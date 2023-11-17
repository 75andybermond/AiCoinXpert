BACKEND=/workspaces/AICoinXpert/src/backend
TEST=/workspaces/AICoinXpert/src/test

launch_docker: # Start database and server . To be executed outside of docker container
	docker compose up
.PHONY: launch_docker

stop_docker: # Stop the running docker container
	docker stop $$(docker ps -a -q)
.PHONY: stop_docker

setup:
	#poetry remove opencv-python
	sudo apt-get update
	sudo apt-get install libgl1-mesa-glx
	sudo apt-get install libglib2.0-0

	sudo apt-get install libcurrand10
	sudo apt-get install libsm6
	sudo apt-get install libgl1-mesa-glx
	sudo apt-get install libglib2.0-dev
	poetry add opencv-python
.PHONY: setup_project

upload_main_data: # If the image is rebuild / need to create first a database ' docker exec -it devcontainer-source-postgres-1 bash ' command is : createdb coins_db
# There is a need to update Minio URL in manually before executing target 'url found in docker compose logs' and update in src/backend/minio_minio.py
	poetry run python src/backend/services.py
.PHONY: upload_main_data

format: # Check format, sort imports and run pylint
	black ${BACKEND}
	isort ${BACKEND}
	pylint ${BACKEND}
.PHONY: format

format_test: # Check format, sort imports and run pylint
	black ${TEST}
	isort ${TEST}
	pylint ${TEST}
.PHONY: format_test

up: # Run the app
	poetry run python ${BACKEND}/app.py
.PHONY: up

backend_test: # Run the backend tests
	pytest -p no:warnings /workspaces/AICoinXpert/src/test/ressources/testsuite/backend/ -v
	lsof -i :5000 # Check if the server is running on port 5000
	kill -9 $$(lsof -t -i:5000) # Kill the server
	make clean_pictures
.PHONY: backend_test

clean_pictures: # Clean the pictures in the tmp folder
	rm -rf /workspaces/AICoinXpert/src/backend/video/tmp/images/*
.PHONY: clean_pictures

# docker compose up
# create table 'coins_db' in postgresql -- command --> ///
# collect minio IP in the docker compose logs and update in src/backend/minio_minio.py
# move models_creation to src/backend and place it back with models.py after the creation of the database
# execute src/backend/services.py

