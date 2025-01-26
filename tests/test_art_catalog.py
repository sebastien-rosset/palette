import pytest
from palette.art_catalog import ArtCatalog


def test_complex_title_extraction():
    filename = "21206-7-2809-1--Ville d'Avray depuis Bois de St-Cloud-Ps-36X36.jpg"
    catalog = ArtCatalog("/dummy/path")
    result = catalog.parse_filename(
        "/dummy/path/" + filename, filename, default_year=2012
    )

    assert result["title"] == "Ville d'Avray depuis Bois de St-Cloud"
    assert result["material"] == "Pastel"
    assert result["width"] == 36.0
    assert result["height"] == 36.0


def test_title_with_comma():
    filename = "21202-1-Les toits roses,Oinville-Ps-35X35.jpg"
    catalog = ArtCatalog("/dummy/path")
    result = catalog.parse_filename(
        "/dummy/path/" + filename, filename, default_year=2012
    )

    assert result["title"] == "Les toits roses,Oinville"
    assert result["material"] == "Pastel"


def test_various_material_indicators():
    catalog = ArtCatalog("/dummy/path")
    test_cases = [
        ("painting-H-36X36.jpg", "Huile"),
        ("painting-Ps-36X36.jpg", "Pastel"),
        ("painting-Ac-36X36.jpg", "Aquarelle"),
        ("painting-Lch-36X36.jpg", "Lavis à l'encre de chine"),
    ]

    for filename, expected_material in test_cases:
        result = catalog.parse_filename("/dummy/path/" + filename, filename)
        assert result["material"] == expected_material


def test_dimension_parsing():
    catalog = ArtCatalog("/dummy/path")
    test_cases = [
        ("painting-36X36.jpg", 36.0, 36.0),
        ("painting-36,5X45,2.jpg", 36.5, 45.2),
        ("painting-36.5X45.2.jpg", 36.5, 45.2),
    ]

    for filename, expected_width, expected_height in test_cases:
        result = catalog.parse_filename("/dummy/path/" + filename, filename)
        assert result["width"] == expected_width
        assert result["height"] == expected_height


def test_title_with_double_dash_before_technical():
    filename = "21205-5-Nostalgie du soir--Ps-50X50.JPG"
    catalog = ArtCatalog("/dummy/path")
    result = catalog.parse_filename(
        "/dummy/path/" + filename, filename, default_year=2012
    )

    assert result["title"] == "Nostalgie du soir"
    assert result["material"] == "Pastel"
    assert result["width"] == 50.0
    assert result["height"] == 50.0
    assert result["catalog_number"] == "21205-5"


def test_title_with_missing_format():
    filename = "21205-4-Coucher de soleil-Ps-50X65--.JPG"
    catalog = ArtCatalog("/dummy/path")
    result = catalog.parse_filename(
        "/dummy/path/" + filename, filename, default_year=2012
    )

    assert result["title"] == "Coucher de soleil"
    assert result["material"] == "Pastel"
    assert result["width"] == 50.0
    assert result["height"] == 65.0
    assert result["catalog_number"] == "21205-4"


def test_title_with_missing_format():
    # Filename with (2) at the end that was added when copying the file.
    filename = "2305-4-Brest, le port de commerce-H-80X40 (2).JPG"
    catalog = ArtCatalog("/dummy/path")
    result = catalog.parse_filename(
        "/dummy/path/" + filename, filename, default_year=2012
    )
    assert result["title"] == "Brest, le port de commerce"
    assert result["width"] == 80.0
    assert result["height"] == 40.0

