def test_smoke_import():
    # Simple import smoke test to ensure package modules are syntactically valid
    import src.python.data_loader as dl

    assert dl.example() == "smoke test OK"
