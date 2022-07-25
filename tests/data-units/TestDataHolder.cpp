
#include <libraries/catch/catch.hpp>

#include <hcorepp/data-units/DataHolder.hpp>

using namespace std;
using namespace hcorepp::dataunits;

void TEST_DATA_HOLDER() {
    SECTION("Data Holder test") {

        int n_rows = 5;
        int n_cols = 5;

        int *data_array = new int[n_rows * n_cols];

        for (size_t i = 0; i < n_rows * n_cols; i++) {
            data_array[i] = i;
        }

        DataHolder<int> data_holder = DataHolder<int>(n_rows, n_cols, 1, data_array);

        REQUIRE(data_holder.GetNumOfRows() == n_rows);
        REQUIRE(data_holder.GetNumOfCols() == n_cols);
        auto data = data_holder.GetData();
        for (size_t i = 0; i < n_rows * n_cols; i++) {
            REQUIRE(data[i] == i);
        }

        delete[] data_array;
    }
}

TEST_CASE("DataHolderTest", "[DATAHOLDER]") {
    TEST_DATA_HOLDER();
}
