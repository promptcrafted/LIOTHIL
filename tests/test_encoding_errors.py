"""Tests for dimljus.encoding.errors — encoding pipeline exceptions."""

from __future__ import annotations

import pytest

from dimljus.encoding.errors import (
    CacheError,
    DimljusEncodingError,
    EncoderError,
    ExpansionError,
)


class TestErrorHierarchy:
    """Tests for error class hierarchy."""

    def test_expansion_is_encoding_error(self) -> None:
        assert issubclass(ExpansionError, DimljusEncodingError)

    def test_cache_is_encoding_error(self) -> None:
        assert issubclass(CacheError, DimljusEncodingError)

    def test_encoder_is_encoding_error(self) -> None:
        assert issubclass(EncoderError, DimljusEncodingError)

    def test_encoding_is_exception(self) -> None:
        assert issubclass(DimljusEncodingError, Exception)


class TestExpansionError:
    def test_message(self) -> None:
        err = ExpansionError("bad frame count")
        assert "Sample expansion failed" in str(err)
        assert "bad frame count" in str(err)
        assert err.detail == "bad frame count"


class TestCacheError:
    def test_message(self) -> None:
        err = CacheError("disk full")
        assert "Cache error" in str(err)
        assert "disk full" in str(err)
        assert err.detail == "disk full"


class TestEncoderError:
    def test_message(self) -> None:
        err = EncoderError("wan_vae", "out of VRAM")
        assert "wan_vae" in str(err)
        assert "out of VRAM" in str(err)
        assert err.encoder_name == "wan_vae"
        assert err.detail == "out of VRAM"
