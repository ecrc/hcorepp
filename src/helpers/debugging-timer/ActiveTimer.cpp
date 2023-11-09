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
        auto new_snapshot_time = std::chrono::high_resolution_clock::now();
        auto end = this->mLastSnapshotTime;
        if (this->mLastSnapshotTimeNamed.count(aSnapshotName) > 0) {
            end = this->mLastSnapshotTimeNamed[aSnapshotName];
            this->mLastSnapshotTimeNamed.erase(aSnapshotName);
        }
        std::chrono::duration<double, std::milli> elapsed_ms = new_snapshot_time - end;
        std::string snapshot_name = aSnapshotName;
        if (this->mSnapshotIndex.count(snapshot_name) > 0) {
            auto index = mSnapshotIndex[snapshot_name];
            this->mSnapshots[index].second += (elapsed_ms.count());
        } else {
            std::pair<std::string, double> entry(snapshot_name, elapsed_ms.count());
            this->mSnapshots.push_back(entry);
            mSnapshotIndex[snapshot_name] = this->mSnapshots.size() - 1;
        }
    }

    void DebuggingTimer::ResetSnapshot(const std::string &aSnapshotName) {
        if (this->mSnapshotIndex.count(aSnapshotName) > 0) {
            auto index = mSnapshotIndex[aSnapshotName];
            this->mSnapshots[index].second = 0;
        }
        this->mLastSnapshotTimeNamed.erase(aSnapshotName);
    }

    void DebuggingTimer::StartSnapshot(const std::string &aSnapshotName) {
        this->mLastSnapshotTimeNamed[aSnapshotName] = std::chrono::high_resolution_clock::now();
    }

    DebuggingTimer *DebuggingTimer::GetDebuggingTimer(size_t aIndex) {
        if (!mDebuggingTimers.empty()) {
            size_t th_idx = 0;
            if (aIndex >= 0) {
                th_idx = aIndex;
            } else {
                th_idx = omp_get_thread_num();
            }
            if (th_idx >= mDebuggingTimers.size()) {
                return mDebuggingTimers[th_idx % mDebuggingTimers.size()];
            }
            if (mDebuggingTimers[th_idx] != nullptr) {
                return mDebuggingTimers[th_idx];
            } else {
                mDebuggingTimers[th_idx] = new DebuggingTimer();
                return mDebuggingTimers[th_idx];
            }
        } else {
            size_t th_idx = 0;
            mDebuggingTimers.push_back(new DebuggingTimer());
            return mDebuggingTimers[th_idx];
        }
    }

    void DebuggingTimer::PrintAllSnapshots(std::ostream &aStream) {
        for (auto &snapshot_pair: mSnapshots) {
            aStream << snapshot_pair.first << ", " << snapshot_pair.second << "\n";
        }
    }

    std::vector<std::string> DebuggingTimer::GetSnapshotsNames() {
        std::vector<std::string> snapshot_names;
        for (auto &snapshot: mSnapshots) {
            snapshot_names.push_back(snapshot.first);
        }
        return snapshot_names;
    }

    void DebuggingTimer::PrintSnapshot(std::string &aName, std::ostream &aStream) const {
        aStream << this->GetSnapshot(aName) << ",";
    }

    double DebuggingTimer::GetSnapshot(const std::string &aSnapshotName) const {
        double time = 0;
        if (this->mSnapshotIndex.count(aSnapshotName) > 0) {
            auto index = this->mSnapshotIndex.find(aSnapshotName)->second;
            time = this->mSnapshots[index].second;
        }
        return time;
    }

    const std::pair<std::string, double> &DebuggingTimer::GetSnapshot(size_t aIndex) const {
        return this->mSnapshots[aIndex];
    }

    size_t DebuggingTimer::GetSnapshotCount() const {
        return this->mSnapshots.size();
    }

    void DebuggingTimer::SetTimersCount(size_t aCount) {
        if (mDebuggingTimers.size() > 0) {
            for (auto &timer: mDebuggingTimers) {
                delete timer;
            }
        }
        mDebuggingTimers.clear();
        mDebuggingTimers.resize(aCount);
        for (auto &timer: mDebuggingTimers) {
            timer = nullptr;
        }
    }

    void DebuggingTimer::ResetAllSnapshots() {
        for (auto &snapshot: mSnapshots) {
            snapshot.second = 0;
        }
        this->mLastSnapshotTimeNamed.clear();
    }
}
