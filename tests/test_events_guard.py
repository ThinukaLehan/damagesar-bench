import pytest

from damagesar_bench.data.events import (
    ConfirmatoryLockError,
    UnknownEventError,
    assert_label_access,
    event_key_from_filename,
    labels_unlocked,
    tile_id_from_filename,
)


def test_event_key_from_bright_filename():
    assert event_key_from_filename("bata-explosion_00000000_post_disaster.tif") == "bata"
    assert event_key_from_filename("la_palma-volcano_00000012_building_damage.tif") == "la_palma"
    assert event_key_from_filename("/x/y/Turkey-earthquake_00000001_post_disaster.tif") == "turkey"


def test_event_key_needs_a_dash():
    with pytest.raises(ValueError):
        event_key_from_filename("nodash_00000001.tif")


def test_tile_id_strips_suffix():
    assert tile_id_from_filename("bata-explosion_00000003_post_disaster.tif", "_post_disaster.tif") == (
        "bata-explosion_00000003"
    )
    with pytest.raises(ValueError):
        tile_id_from_filename("bata-explosion_00000003_pre_disaster.tif", "_post_disaster.tif")


def test_confirmatory_labels_locked_before_freeze(events_cfg, before_freeze):
    with pytest.raises(ConfirmatoryLockError):
        assert_label_access("haiti", events_cfg, today=before_freeze)


def test_development_and_secondary_always_allowed(events_cfg, before_freeze):
    assert_label_access("beirut", events_cfg, today=before_freeze)
    assert_label_access("marshall", events_cfg, today=before_freeze)


def test_confirmatory_unlocked_on_and_after_freeze(events_cfg, after_freeze):
    assert labels_unlocked(events_cfg, today=events_cfg.freeze_date)
    assert_label_access("haiti", events_cfg, today=after_freeze)


def test_env_unlock(events_cfg, before_freeze, monkeypatch):
    monkeypatch.setenv("DAMAGESAR_UNLOCK_CONFIRMATORY", "1")
    assert_label_access("haiti", events_cfg, today=before_freeze)


def test_unknown_event(events_cfg, before_freeze):
    with pytest.raises(UnknownEventError):
        assert_label_access("atlantis", events_cfg, today=before_freeze)
