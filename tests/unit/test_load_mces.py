"""Tests for simba/load_mces/load_mces.py"""

import numpy as np

from simba.core.chemistry.mces_loader.load_mces import LoadMCES


class TestLoadMCES:
    def test_normalize_ed_basic(self):
        ed = np.array([0.0, 1.0, 2.5, 5.0])
        result = LoadMCES.normalize_ed(ed, max_ed=5)

        expected = np.array([1.0, 0.8, 0.5, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_ed_exceeds_max(self):
        ed = np.array([0.0, 3.0, 7.0, 10.0])
        result = LoadMCES.normalize_ed(ed, max_ed=5)

        assert result[0] == 1.0
        assert result[1] == 0.4
        assert result[2] == 0.0
        assert result[3] == 0.0

    def test_normalize_ed_all_zeros(self):
        ed = np.array([0.0, 0.0, 0.0])
        result = LoadMCES.normalize_ed(ed, max_ed=5)

        np.testing.assert_array_equal(result, np.array([1.0, 1.0, 1.0]))

    def test_normalize_mces20_basic(self):
        mcs20 = np.array([0.0, 5.0, 10.0, 20.0])
        result = LoadMCES.normalize_mces20(mcs20, max_value=20.0)

        expected = np.array([1.0, 0.75, 0.5, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_mces20_with_negative_removal(self):
        mcs20 = np.array([0.0, 10.0, 25.0, 30.0])
        result = LoadMCES.normalize_mces20(
            mcs20, max_value=20.0, remove_negative_values=True
        )

        assert result[0] == 1.0
        assert result[1] == 0.5
        assert result[2] == 0.0
        assert result[3] == 0.0

    def test_normalize_mces20_without_negative_removal(self):
        mcs20 = np.array([0.0, 10.0, 25.0])
        result = LoadMCES.normalize_mces20(
            mcs20, max_value=20.0, remove_negative_values=False
        )

        assert result[0] == 1.0
        assert result[1] == 0.5
        assert result[2] < 0

    def test_find_file(self, tmp_path):
        test_dir = tmp_path / "test_data"
        test_dir.mkdir()

        file1 = test_dir / "prefix_node0_chunk0.npy"
        file2 = test_dir / "prefix_node1_chunk0.npy"
        file3 = test_dir / "other_file.npy"

        file1.touch()
        file2.touch()
        file3.touch()

        result = LoadMCES.find_file(str(test_dir), "prefix")

        assert len(result) == 2
        assert any("prefix_node0_chunk0.npy" in path for path in result)
        assert any("prefix_node1_chunk0.npy" in path for path in result)
        assert not any("other_file.npy" in path for path in result)

    def test_find_file_no_matches(self, tmp_path):
        test_dir = tmp_path / "test_data"
        test_dir.mkdir()

        file1 = test_dir / "other_file.npy"
        file1.touch()

        result = LoadMCES.find_file(str(test_dir), "prefix")

        assert len(result) == 0

    def test_find_file_nested_directories(self, tmp_path):
        test_dir = tmp_path / "test_data"
        test_dir.mkdir()

        subdir = test_dir / "subdir"
        subdir.mkdir()

        file1 = test_dir / "prefix_node0_chunk0.npy"
        file2 = subdir / "prefix_node1_chunk0.npy"

        file1.touch()
        file2.touch()

        result = LoadMCES.find_file(str(test_dir), "prefix")

        assert len(result) == 2

    def test_remove_excess_low_pairs_basic(self):
        data = np.array(
            [
                [0, 1, 6.0],
                [0, 2, 7.0],
                [1, 2, 8.0],
                [2, 3, 9.0],
                [3, 4, 10.0],
                [4, 5, 2.0],
                [5, 6, 3.0],
                [6, 7, 4.0],
            ]
        )

        result = LoadMCES.remove_excess_low_pairs(
            data, remove_percentage=0.5, max_value=5, target_column=2
        )

        assert result.shape[0] <= data.shape[0]
        assert result.shape[1] == data.shape[1]

        high_pairs = result[result[:, 2] < 5]
        assert len(high_pairs) == 3

    def test_add_high_similarity_pairs_edit_distance(self):
        merged_array = np.array(
            [
                [0, 1, 0.5],
                [1, 2, 0.6],
                [2, 3, 0.7],
            ]
        )

        result = LoadMCES.add_high_similarity_pairs_edit_distance(merged_array)

        assert result.shape[0] > merged_array.shape[0]
        assert result.shape[1] == merged_array.shape[1]

        identical_pairs = result[result[:, 0] == result[:, 1]]
        assert len(identical_pairs) > 0
        np.testing.assert_array_equal(
            identical_pairs[:, 2], np.ones(len(identical_pairs))
        )

    def test_load_raw_data_with_files(self, tmp_path):
        test_dir = tmp_path / "test_data"
        test_dir.mkdir()

        data1 = np.array([[0, 1, 0.5], [1, 2, 0.6]])
        data2 = np.array([[2, 3, 0.7], [3, 4, 0.8]])

        file1 = test_dir / "prefix_node0_chunk0.npy"
        file2 = test_dir / "prefix_node1_chunk0.npy"

        np.save(file1, data1)
        np.save(file2, data2)

        result = LoadMCES.load_raw_data(str(test_dir), "prefix")

        assert result.shape[0] == 4
        assert result.shape[1] == 3

    def test_load_raw_data_empty_directory(self, tmp_path):
        test_dir = tmp_path / "empty_data"
        test_dir.mkdir()

        result = LoadMCES.load_raw_data(str(test_dir), "prefix")

        assert result.shape[0] == 0

    def test_load_raw_data_with_partitions_limit(self, tmp_path):
        test_dir = tmp_path / "test_data"
        test_dir.mkdir()

        data1 = np.array([[0, 1, 0.5]])
        data2 = np.array([[1, 2, 0.6]])
        data3 = np.array([[2, 3, 0.7]])

        np.save(test_dir / "prefix_node0_chunk0.npy", data1)
        np.save(test_dir / "prefix_node1_chunk0.npy", data2)
        np.save(test_dir / "prefix_node2_chunk0.npy", data3)

        result = LoadMCES.load_raw_data(str(test_dir), "prefix", partitions=2)

        assert result.shape[0] == 2
        assert result.shape[1] == 3

    def test_remove_excess_low_pairs_no_removal(self):
        data = np.array(
            [
                [0, 1, 6.0],
                [0, 2, 7.0],
                [1, 2, 8.0],
            ]
        )

        result = LoadMCES.remove_excess_low_pairs(
            data, remove_percentage=0.0, max_value=5, target_column=2
        )

        assert result.shape[0] == data.shape[0]

    def test_normalize_ed_with_custom_max(self):
        ed = np.array([0.0, 2.5, 5.0, 10.0])
        result = LoadMCES.normalize_ed(ed, max_ed=10)

        expected = np.array([1.0, 0.75, 0.5, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_add_high_similarity_pairs_with_extra_column(self):
        merged_array = np.array(
            [
                [0, 1, 0.5, 0.6],
                [1, 2, 0.6, 0.7],
                [2, 3, 0.7, 0.8],
            ]
        )

        result = LoadMCES.add_high_similarity_pairs_edit_distance(merged_array)

        assert result.shape[0] > merged_array.shape[0]
        assert result.shape[1] == 4

        identical_pairs = result[result[:, 0] == result[:, 1]]
        assert len(identical_pairs) > 0
        np.testing.assert_array_equal(
            identical_pairs[:, 2], np.ones(len(identical_pairs))
        )
        np.testing.assert_array_equal(
            identical_pairs[:, 3], np.ones(len(identical_pairs))
        )

    def test_load_mces_20_data(self, tmp_path):
        base_dir = tmp_path / "mces_data"

        folder0 = base_dir / "0"
        folder1 = base_dir / "1"
        folder2 = base_dir / "2"

        folder0.mkdir(parents=True)
        folder1.mkdir(parents=True)
        folder2.mkdir(parents=True)

        data0 = np.array([[0, 1, 0.5], [1, 2, 0.6]])
        data1 = np.array([[2, 3, 0.7]])
        data2 = np.array([])

        np.save(folder0 / "mces_node0_chunk0.npy", data0)
        np.save(folder1 / "mces_node0_chunk0.npy", data1)
        np.save(folder2 / "mces_node0_chunk0.npy", data2)

        result = LoadMCES.load_mces_20_data(
            str(base_dir) + "/", "mces", number_folders=3
        )

        assert result.shape[0] == 3
        assert result.shape[1] == 3
