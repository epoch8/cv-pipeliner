black:
	autoflake -r --in-place --remove-all-unused-imports cv_pipeliner/ tests/ --exclude cv_pipeliner/__init__.py
	black --verbose --config black.toml cv_pipeliner/ tests/
