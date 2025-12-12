import numpy as np
import gtsam
import math

GRAVITY = 9.807


class MultiSensorState:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.cur_t = 0.0

        """ IMU-centered states """
        self.timestamps = []  # timestamps (len == N)

        self.wTbs = []  # poses      (len == N)
        self.vs = []  # vels       (len == N)
        self.bs = []  # biases     (len == N)

        self.preintegrations = []  # preintegrations (len == N)
        self.preintegrations_meas = []  # raw IMU data    (len == N)
        self.preintegration_temp = None  # used for high-frequency prediction
        self.pose_temp = None  # used for high-frequency prediction

        self.gnss_valid = []  # GNSS flags (len == N)
        self.gnss_position = []  # GNSS pos   (len == N)

        self.odo_valid = []  # Odo flags  (len == N)
        self.odo_vel = []  # Odo vel    (len == N)

        self.marg_factor = None
        self.set_imu_params()

        # IMU loosening controls
        self.imu_gap_threshold = (
            getattr(cfg, "IMU_GAP_THRESHOLD", 0.025) if cfg is not None else 0.025
        )
        self.low_motion_loosen = (
            bool(getattr(cfg, "IMU_LOOSE_WITH_LOW_MOTION", False))
            if cfg is not None
            else False
        )
        self.low_motion_vel_thresh = (
            float(getattr(cfg, "IMU_LOOSE_LOW_VEL_THRESH", 0.05))
            if cfg is not None
            else 0.05
        )
        self.low_motion_duration = (
            float(getattr(cfg, "IMU_LOOSE_LOW_VEL_DURATION", 0.5))
            if cfg is not None
            else 0.5
        )
        self.low_motion_exit_ratio = (
            float(getattr(cfg, "IMU_LOOSE_LOW_VEL_EXIT_RATIO", 2.0))
            if cfg is not None
            else 2.0
        )
        if self.low_motion_exit_ratio < 1.0:
            self.low_motion_exit_ratio = 1.0

        self.low_accel_loosen = (
            bool(getattr(cfg, "IMU_LOOSE_WITH_LOW_ACCEL", False))
            if cfg is not None
            else False
        )
        self.low_accel_thresh = (
            float(getattr(cfg, "IMU_LOOSE_LOW_ACCEL_THRESH", 0.05))
            if cfg is not None
            else 0.05
        )
        self.low_accel_duration = (
            float(getattr(cfg, "IMU_LOOSE_LOW_ACCEL_DURATION", 0.5))
            if cfg is not None
            else 0.5
        )
        self.low_accel_exit_ratio = (
            float(getattr(cfg, "IMU_LOOSE_LOW_ACCEL_EXIT_RATIO", 2.0))
            if cfg is not None
            else 2.0
        )
        if self.low_accel_exit_ratio < 1.0:
            self.low_accel_exit_ratio = 1.0
        self.low_accel_min_speed = (
            float(getattr(cfg, "IMU_LOOSE_LOW_ACCEL_MIN_SPEED", 0.5))
            if cfg is not None
            else 0.5
        )

        self._low_motion_start_time = None
        self._low_motion_active = False
        self._low_accel_start_time = None
        self._low_accel_active = False
        self._active_preintegration_is_loose = False
        self._last_velocity = None
        self._last_velocity_time = None

    def set_imu_params(self, noise=None):
        # default
        accel_noise_sigma = 0.0
        gyro_noise_sigma = 0.0
        accel_bias_rw_sigma = 0.0
        gyro_bias_rw_sigma = 0.0

        if noise is not None:
            accel_noise_sigma = noise[0]
            gyro_noise_sigma = noise[1]
            accel_bias_rw_sigma = noise[2]
            gyro_bias_rw_sigma = noise[3]

        measured_acc_cov = np.eye(3, 3) * math.pow(accel_noise_sigma, 2)
        measured_omega_cov = np.eye(3, 3) * math.pow(gyro_noise_sigma, 2)
        integration_error_cov = np.eye(3, 3) * 0e-8
        bias_acc_cov = np.eye(3, 3) * math.pow(accel_bias_rw_sigma, 2)
        bias_omega_cov = np.eye(3, 3) * math.pow(gyro_bias_rw_sigma, 2)

        params = gtsam.PreintegrationCombinedParams.MakeSharedU(GRAVITY)
        params.setAccelerometerCovariance(measured_acc_cov)
        params.setIntegrationCovariance(integration_error_cov)
        params.setGyroscopeCovariance(measured_omega_cov)
        params.setBiasAccCovariance(bias_acc_cov)
        params.setBiasOmegaCovariance(bias_omega_cov)
        self.params = params

        params_loose = gtsam.PreintegrationCombinedParams.MakeSharedU(GRAVITY)
        params_loose.setAccelerometerCovariance(measured_acc_cov * 100)
        params_loose.setIntegrationCovariance(integration_error_cov)
        params_loose.setGyroscopeCovariance(measured_omega_cov * 100)
        params_loose.setBiasAccCovariance(bias_acc_cov)
        params_loose.setBiasOmegaCovariance(bias_omega_cov)
        self.params_loose = params_loose

    def init_first_state(self, t, pos, R, vel):
        self.timestamps.append(t)
        self.wTbs.append(gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(pos)))
        self.vs.append(vel)
        self.bs.append(
            gtsam.imuBias.ConstantBias(
                np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])
            )
        )
        self.preintegrations.append(
            gtsam.PreintegratedCombinedMeasurements(self.params, self.bs[-1])
        )
        self.preintegrations_meas.append([])
        self.preintegration_temp = gtsam.PreintegratedCombinedMeasurements(
            self.params, self.bs[-1]
        )
        self._active_preintegration_is_loose = False
        self.gnss_valid.append(False)
        self.gnss_position.append(np.array([0.0, 0.0, 0.0]))
        self.odo_valid.append(False)
        self.odo_vel.append(np.array([0.0, 0.0, 0.0]))

        self.cur_t = t

    def append_imu(self, t, measuredAcc, measuredOmega):
        dt = t - self.cur_t
        if dt > 0:
            if dt > self.imu_gap_threshold:
                if not self._active_preintegration_is_loose:
                    print("IMU gap detected; loosening preintegration noise.")
                self._loosen_active_preintegration()
            self.preintegrations[-1].integrateMeasurement(
                measuredAcc, measuredOmega, dt
            )
        if dt < 0:
            raise Exception("may not happen")
        self.preintegrations_meas[-1].append([measuredAcc, measuredOmega, dt, t])
        # print('append_imu: ',measuredAcc,measuredOmega,t - self.cur_t,t)
        self.last_measuredAcc = measuredAcc
        self.last_measuredOmega = measuredOmega
        self.cur_t = t

    def append_imu_temp(self, t, measuredAcc, measuredOmega, predict_pose=False):
        if t - self.cur_t > 0:
            self.preintegration_temp.integrateMeasurement(
                measuredAcc, measuredOmega, t - self.cur_t
            )
        if predict_pose:
            prev_state = gtsam.gtsam.NavState(self.wTbs[-1], self.vs[-1])
            prev_bias = self.bs[-1]
            self.pose_temp = self.preintegration_temp.predict(prev_state, prev_bias)

    def append_img(self, t):
        self.cur_t = t
        prev_state = gtsam.gtsam.NavState(self.wTbs[-1], self.vs[-1])
        prev_bias = self.bs[-1]
        prop_state = self.preintegrations[-1].predict(prev_state, prev_bias)
        if self.preintegrations[-1].deltaTij() > 1.0:
            prop_state = gtsam.gtsam.NavState(self.wTbs[-1], self.vs[-1])

        prop_velocity = np.asarray(prop_state.velocity(), dtype=np.float64)
        speed = float(np.linalg.norm(prop_velocity))

        accel_mag = None
        if self._last_velocity is not None and self._last_velocity_time is not None:
            dt_vel = t - self._last_velocity_time
            if dt_vel > 0:
                delta_v = prop_velocity - self._last_velocity
                accel_mag = float(np.linalg.norm(delta_v) / dt_vel)

        low_motion_active = self._should_loosen_for_low_motion(t, speed)
        low_accel_active = self._should_loosen_for_low_accel(t, speed, accel_mag)
        #print(t, speed, accel_mag, low_motion_active, low_accel_active)
        if low_accel_active:
            self._loosen_active_preintegration()
            print("loosed")

        self._last_velocity = prop_velocity.copy()
        self._last_velocity_time = t

        self.timestamps.append(t)
        self.wTbs.append(prop_state.pose())
        self.vs.append(prop_state.velocity())
        self.bs.append(prev_bias)
        self.gnss_valid.append(False)  # 不用gps因此都为False
        self.gnss_position.append(np.array([0.0, 0.0, 0.0]))
        self.odo_valid.append(False)  # 不用里程计，因此都为False
        self.odo_vel.append(np.array([0.0, 0.0, 0.0]))

        loosen_enabled = self._low_motion_active or self._low_accel_active
        next_params = self.params_loose if loosen_enabled else self.params
        self.preintegrations.append(
            gtsam.PreintegratedCombinedMeasurements(next_params, self.bs[-1])
        )
        self.preintegrations_meas.append([])
        self.preintegration_temp = gtsam.PreintegratedCombinedMeasurements(
            next_params, self.bs[-1]
        )
        self._active_preintegration_is_loose = next_params is self.params_loose

    # ugly implementation
    # this should be called after append_img()
    def append_gnss(self, t, pos):
        if math.fabs(self.cur_t - t) > 0.01:
            print("Skip GNSS data due to unsynchronization!!")
        else:
            self.gnss_valid[-1] = True
            self.gnss_position[-1] = pos

    def append_odo(self, t, vel):
        if math.fabs(self.cur_t - t) > 0.01:
            print("Skip ODO data due to unsynchronization!!")
        else:
            self.odo_valid[-1] = True
            self.odo_vel[-1] = vel

    def predict(self):
        prev_state = gtsam.gtsam.NavState(self.wTbs[-1], self.vs[-1])
        prev_bias = self.bs[-1]
        self.preintegrations[-1].predict(prev_state, prev_bias)

    def _should_loosen_for_low_motion(self, t, speed):
        if not self.low_motion_loosen:
            self._low_motion_start_time = None
            self._low_motion_active = False
            return False

        if speed < self.low_motion_vel_thresh:
            if self._low_motion_start_time is None:
                self._low_motion_start_time = t
            elif (t - self._low_motion_start_time) >= self.low_motion_duration:
                self._low_motion_active = True
        else:
            exit_speed = self.low_motion_vel_thresh * self.low_motion_exit_ratio
            if self._low_motion_active and speed < exit_speed:
                # remain in low-motion mode until we clearly leave the dead-zone
                pass
            else:
                self._low_motion_active = False
                self._low_motion_start_time = None

        return self._low_motion_active

    def _should_loosen_for_low_accel(self, t, speed, accel_mag):
        if not self.low_accel_loosen:
            self._low_accel_start_time = None
            self._low_accel_active = False
            return False

        valid_speed = speed is not None and speed >= self.low_accel_min_speed
        if accel_mag is None or not valid_speed:
            self._low_accel_start_time = None
            self._low_accel_active = False
            return False

        if accel_mag < self.low_accel_thresh:
            if self._low_accel_start_time is None:
                self._low_accel_start_time = t
            elif (t - self._low_accel_start_time) >= self.low_accel_duration:
                self._low_accel_active = True
                print("low accel active")
        else:
            exit_accel = self.low_accel_thresh * self.low_accel_exit_ratio
            if self._low_accel_active and accel_mag < exit_accel:
                pass
            else:
                self._low_accel_active = False
                self._low_accel_start_time = None

        return self._low_accel_active

    def _loosen_active_preintegration(self):
        if not self.preintegrations:
            return
        if not self.preintegrations_meas:
            return
        if self._active_preintegration_is_loose:
            return

        pim = gtsam.PreintegratedCombinedMeasurements(self.params_loose, self.bs[-1])
        for measuredAcc, measuredOmega, dt, _ in self.preintegrations_meas[-1]:
            if dt > 0:
                pim.integrateMeasurement(measuredAcc, measuredOmega, dt)

        self.preintegrations[-1] = pim
        self._active_preintegration_is_loose = True
