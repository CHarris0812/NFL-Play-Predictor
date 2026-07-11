import features
import ingestion
import models
import serving


def test_packages_importable():
    assert ingestion and features and models and serving
