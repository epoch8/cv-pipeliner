black:
	autoflake -r --in-place --remove-all-unused-imports cv_pipeliner/ tests/ --exclude cv_pipeliner/__init__.py
	isort --profile black cv_pipeliner/ tests/
	black --verbose --config black.toml cv_pipeliner/ tests/
