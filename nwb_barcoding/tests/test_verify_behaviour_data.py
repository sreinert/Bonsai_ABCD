from nwb_barcoding.verify_behaviour_data import lick_count_mismatch_is_material


def test_single_lick_cross_logger_difference_is_not_material_for_long_session():
    assert not lick_count_mismatch_is_material(7651, 7652)


def test_large_lick_cross_logger_difference_is_material():
    assert lick_count_mismatch_is_material(7651, 4443)


def test_single_lick_difference_is_material_for_small_count():
    assert lick_count_mismatch_is_material(10, 11)
