#include "kalman_tracker.hpp"

#include <stdexcept>

namespace hw
{
    KalmanTracker::KalmanTracker() = default;
    
    bool KalmanTracker::isTracking() const
    {
        return tracking_;
    }

    void KalmanTracker::reset()
    {
        tracking_ = false;
        x_ = AxisFilter{};
        y_ = AxisFilter{};
        z_ = AxisFilter{};
    }

    void KalmanTracker::AxisFilter::reset(double measured_position)
    {
        position = measured_position;
        velocity = 0.0;
        p00 = 1.0;
        p01 = 0.0;
        p10 = 0.0;
        p11 = 1.0;
    }

    void KalmanTracker::AxisFilter::predict(double dt, double process_noise)
    {
        dt = std::max(dt, 0.0);

        position = position + velocity * dt;

        double q00 = process_noise * (dt * dt * dt * dt / 4.0);
        double q01 = process_noise * (dt * dt * dt / 2.0);
        double q10 = q01;
        double q11 = process_noise * (dt * dt);

        double f_p_ft_00 = p00 + dt * (p10 + p01 + dt * p11);
        double f_p_ft_01 = p01 + dt * p11;
        double f_p_ft_10 = p10 + dt * p11;
        double f_p_ft_11 = p11;

        p00 = f_p_ft_00 + q00;
        p01 = f_p_ft_01 + q01;
        p10 = f_p_ft_10 + q10;
        p11 = f_p_ft_11 + q11;
    }

    void KalmanTracker::AxisFilter::update(double measured_position, double measurement_noise)
    {
        double residual = measured_position - position;

        double S = p00 + measurement_noise;

        if (S <= 0.0) {
            return;
        }

        double k0 = p00 / S;
        double k1 = p10 / S;

        position = position + k0 * residual;
        velocity = velocity + k1 * residual;

        double next_p00 = (1.0 - k0) * p00;
        double next_p01 = (1.0 - k0) * p01;
        double next_p10 = -k1 * p00 + p10;
        double next_p11 = -k1 * p01 + p11;

        p00 = next_p00;
        p01 = next_p01;
        p10 = next_p10;
        p11 = next_p11;
    }

    TrackState KalmanTracker::update(const Vec3 &measurement, double dt)
    {
       if (!tracking_) {
            x_.reset(measurement.x);
            y_.reset(measurement.y);
            z_.reset(measurement.z);
            tracking_ = true;
            return stateFromFilters();
        }

        x_.predict(dt, process_noise_);
        y_.predict(dt, process_noise_);
        z_.predict(dt, process_noise_);

        x_.update(measurement.x, measurement_noise_);
        y_.update(measurement.y, measurement_noise_);
        z_.update(measurement.z, measurement_noise_);

        return stateFromFilters();
    }

    TrackState KalmanTracker::predict(double dt)
    {
        if (!tracking_) {
            return TrackState{false, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        }

        x_.predict(dt, process_noise_);
        y_.predict(dt, process_noise_);
        z_.predict(dt, process_noise_);

        return stateFromFilters();
    }

    TrackState KalmanTracker::stateFromFilters() const
    {
        return {
            true,
            {x_.position, y_.position, z_.position},
            {x_.velocity, y_.velocity, z_.velocity},
        };
    }
} // namespace hw
