embed:
	docker compose run --rm app python training/embed.py

train:
	docker compose run --rm app python training/train.py

pipeline: embed train

dashboard:
	docker compose up dashboard

predict:
	docker compose run --rm app python prediction/predict.py --text "$(TEXT)"
