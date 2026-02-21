embed:
	docker compose run --rm app python training/embed.py

train:
	docker compose run --rm app python training/train.py

pipeline: embed train

dashboard:
	docker compose up dashboard

predict:
	docker compose run --rm app python prediction/predict.py --text "$(TEXT)"

build_fsi:
	docker compose run --rm app python src/data/build_fsi_target.py \
		--start $(or $(FSI_START),2023-01-01) --end $(or $(FSI_END),2024-12-31)

seed_fsi:
	docker compose run --rm app python db/seed_fsi.py

backfill:
	docker compose run --rm app python src/ingestion/historical_backfill.py \
		--date-from $(DATE_FROM) --date-to $(DATE_TO)

daily_pipeline:
	docker compose run --rm app python src/ingestion/daily_pipeline.py
