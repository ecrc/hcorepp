/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include <hcorepp/helpers/Timer.hpp>

namespace hcorepp {
    namespace helpers {

        Timer::Timer() {
            this->StartSnapshot();
        }

        void Timer::StartSnapshot() {
            this->mLastSnapshotTime = std::chrono::high_resolution_clock::now();
        }

        void Timer::Snapshot(const std::string &aSnapshotName) {
            auto new_snapshot_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed_ms = new_snapshot_time - this->mLastSnapshotTime;
            std::string snapshot_name = aSnapshotName;
            if (this->mSnapshotIndex.count(snapshot_name) > 0) {
                auto index = mSnapshotIndex[snapshot_name];
                this->mSnapshots[index].second += (elapsed_ms.count());
            } else {
                std::pair<std::string, double> entry(snapshot_name, elapsed_ms.count());
                this->mSnapshots.push_back(entry);
                mSnapshotIndex[snapshot_name] = this->mSnapshots.size() - 1;
            }
            this->StartSnapshot();
        }

        void Timer::ResetSnapshot(const std::string &aSnapshotName) {
            if (this->mSnapshotIndex.count(aSnapshotName) > 0) {
                auto index = mSnapshotIndex[aSnapshotName];
                this->mSnapshots[index].second = 0;
            }
        }

    }//namespace helpers
}//namespace hcorepp
