/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include <hcorepp/helpers/DebuggingTimer.hpp>
#include <iostream>
#include <omp.h>

namespace hcorepp::helpers {
    std::vector<hcorepp::helpers::DebuggingTimer *> hcorepp::helpers::DebuggingTimer::mDebuggingTimers;

    DebuggingTimer::DebuggingTimer() {
    }

    void DebuggingTimer::Snapshot(const std::string &aSnapshotName) {
    }

    void DebuggingTimer::ResetSnapshot(const std::string &aSnapshotName) {
    }

    void DebuggingTimer::StartSnapshot(const std::string &aSnapshotName) {
    }

    DebuggingTimer *DebuggingTimer::GetDebuggingTimer(size_t aIndex) {
        static auto *timer = new DebuggingTimer();
        return timer;
    }

    void DebuggingTimer::PrintAllSnapshots(std::ostream &aStream) {
    }

    std::vector<std::string> DebuggingTimer::GetSnapshotsNames() {
    }

    void DebuggingTimer::PrintSnapshot(std::string &aName, std::ostream &aStream) const {
    }

    double DebuggingTimer::GetSnapshot(const std::string &aSnapshotName) const {
    }

    const std::pair<std::string, double> &DebuggingTimer::GetSnapshot(size_t aIndex) const {
    }

    size_t DebuggingTimer::GetSnapshotCount() const {
    }

    void DebuggingTimer::SetTimersCount(size_t aCount) {
    }

    void DebuggingTimer::ResetAllSnapshots() {
    }
}
