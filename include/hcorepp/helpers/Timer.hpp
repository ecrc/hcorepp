/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_HELPERS_TIMER_HPP
#define HCOREPP_HELPERS_TIMER_HPP

#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>

namespace hcorepp {
    namespace helpers {
        /**
         * @brief
         * Timer class to help with benchmarking and timing examples.
         */
        class Timer {
        public:
            /**
             * @brief
             * Default Constructor.
             */
            Timer();

            /**
             * @brief
             * Take a snapshot with a given name, this will assign all the time
             * from the last call of this function or the constructor, whichever
             * closest, to the given name.
             *
             * @param[in] aSnapshotName
             * Name given to the event snapshot.
             */
            void Snapshot(const std::string &aSnapshotName);

            /**
             * @brief
             * Resets the time associated with a given snapshot name.
             *
             * @param[in] aSnapshotName
             * Name given to the event snapshot.
             */
            void ResetSnapshot(const std::string &aSnapshotName);

            /**
             * @brief
             * Start the snapshot timing.
             */
            void StartSnapshot(const std::string& aSnapshotName = "");

            /**
             * @brief
             * Get the total number of snapshots taken by the timer.
             *
             * @return
             * Number of snapshots.
             */
            size_t GetSnapshotCount() const {
                return this->mSnapshots.size();
            }

            /**
             * @brief
             * Retrieve a snapshot information from the timer.
             *
             * @param[in] aIndex
             * Index of the snapshot to retrieve.
             *
             * @return
             * A pair of the snapshot name and the time associated with it.
             */
            const std::pair<std::string, double> &GetSnapshot(size_t aIndex) const {
                return this->mSnapshots[aIndex];
            }

            /**
             * @brief
             * Retrieve a snapshot time from the timer.
             *
             * @param[in] aSnapshotName
             * Name of the target snapshot
             *
             * @return
             * The time associated with the snapshot, if it is not existent,
             * it will return 0;
             */
            double GetSnapshot(const std::string &aSnapshotName) const {
                double time = 0;
                if (this->mSnapshotIndex.count(aSnapshotName) > 0) {
                    auto index = this->mSnapshotIndex.find(aSnapshotName)->second;
                    time = this->mSnapshots[index].second;
                }
                return time;
            }

            /**
             * @brief
             * Default destructor.
             */
            ~Timer() = default;

        private:
            /// The time of the last snapshot taken by the timer.
            std::chrono::time_point<std::chrono::system_clock> mLastSnapshotTime;
            /// Vector of all the snapshots taken and their names.
            std::vector<std::pair<std::string, double>> mSnapshots;
            /// Map between snapshot name and index to be able to accumulate timings.
            std::unordered_map<std::string, size_t> mSnapshotIndex;
        };
    }//namespace helpers
}//namespace hcorepp

#endif //HCOREPP_HELPERS_TIMER_HPP
