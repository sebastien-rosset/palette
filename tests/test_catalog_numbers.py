import pytest
from palette.art_catalog import ArtCatalog


def test_pre_2000_basic():
    """Test basic pre-2000 catalog numbers"""
    catalog = ArtCatalog("/dummy/path")
    result = catalog.parse_catalog_number("702-1")
    assert result == {"year": 1997, "month": 2, "item_number": 1}


def test_post_2000_basic():
    """Test basic post-2000 catalog numbers"""
    catalog = ArtCatalog("/dummy/path")
    result = catalog.parse_catalog_number("2603-5")
    assert result == {"year": 2006, "month": 3, "item_number": 5}


def test_invalid_month():
    """Test catalog numbers with invalid months"""
    catalog = ArtCatalog("/dummy/path")
    assert catalog.parse_catalog_number("713-1") is None  # Month 13 invalid
    assert catalog.parse_catalog_number("2600-1") is None  # Month 00 invalid
    assert catalog.parse_catalog_number("2613-1") is None  # Month 13 invalid


def test_invalid_formats():
    """Test various invalid formats"""
    invalid_cases = [
        "",  # Empty string
        "123",  # No hyphen
        "123-",  # No item number
        "-123",  # No main number
        "12345-1",  # Too many digits
        "1a3-1",  # Non-numeric characters
        "123-a",  # Non-numeric item number
        None,  # None input
    ]
    catalog = ArtCatalog("/dummy/path")
    for case in invalid_cases:
        assert catalog.parse_catalog_number(case) is None


def test_edge_months():
    """Test edge cases for months"""
    # Valid months
    catalog = ArtCatalog("/dummy/path")
    assert catalog.parse_catalog_number("701-1")["month"] == 1  # January
    assert catalog.parse_catalog_number("712-1")["month"] == 12  # December
    assert catalog.parse_catalog_number("2601-1")["month"] == 1  # January
    assert catalog.parse_catalog_number("2612-1")["month"] == 12  # December


def test_with_additional_parts():
    """Test catalog numbers with additional parts after item number"""
    catalog = ArtCatalog("/dummy/path")
    result = catalog.parse_catalog_number("2603-5-ex22308-4")
    assert result == {"year": 2006, "month": 3, "item_number": 5}


def test_all_valid_months():
    """Test all valid months for both pre and post 2000"""
    catalog = ArtCatalog("/dummy/path")
    for month in range(1, 13):
        # Pre-2000
        month_str = f"{month:02d}"
        result = catalog.parse_catalog_number(f"7{month_str}-1")
        assert result["month"] == month
        assert result["year"] == 1997

        # Post-2000
        result = catalog.parse_catalog_number(f"26{month_str}-1")
        assert result["month"] == month
        assert result["year"] == 2006


def test_different_years():
    """Test different years in both centuries"""
    # Pre-2000
    catalog = ArtCatalog("/dummy/path")
    for year_digit in range(10):
        result = catalog.parse_catalog_number(f"{year_digit}01-1")
        assert result["year"] == 1990 + year_digit

    # Post-2000
    for year_digit in range(10):
        result = catalog.parse_catalog_number(f"2{year_digit}01-1")
        assert result["year"] == 2000 + year_digit


def test_item_numbers():
    catalog = ArtCatalog("/dummy/path")
    """Test various item numbers"""
    # Single digit
    assert catalog.parse_catalog_number("701-1")["item_number"] == 1

    # Multiple digits
    assert catalog.parse_catalog_number("701-42")["item_number"] == 42
    assert catalog.parse_catalog_number("2601-123")["item_number"] == 123


def test_invalid_century_indicator():
    catalog = ArtCatalog("/dummy/path")
    """Test invalid century indicators for post-2000 format"""
    assert (
        catalog.parse_catalog_number("3601-1") is None
    )  # Century indicator should be 2
    assert (
        catalog.parse_catalog_number("1601-1") is None
    )  # Century indicator should be 2


if __name__ == "__main__":
    pytest.main([__file__])
