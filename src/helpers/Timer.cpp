/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing and Research Center Property       *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#include <hcorepp/helpers/Timer.hpp>

namespace hcorepp {
    namespace helpers {

        Timer::Timer() {
            this->StartSnapshot();
        }

        void Timer::StartSnapshot() {
            this->mLastSnapshotTime = std::chrono::system_clock::now();
        }

        void Timer::Snapshot(const std::string &aSnapshotName) {
            auto new_snapshot_time = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = new_snapshot_time - this->mLastSnapshotTime;
            if (this->mSnapshotIndex.count(aSnapshotName) > 0) {
                auto index = mSnapshotIndex[aSnapshotName];
                this->mSnapshots[index].second += elapsed_seconds.count();
            } else {
                this->mSnapshots.emplace_back(aSnapshotName, elapsed_seconds.count());
                mSnapshotIndex[aSnapshotName] = this->mSnapshots.size() - 1;
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