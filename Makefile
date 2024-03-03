# Description: Makefile for setting up the project environment and running the API server

# Virtual environment setup
create_env:
	python3 -m venv .venv;
	@echo "Virtual environment created. Activate it using: 'source .venv/bin/activate'";
	@echo "After activation, run 'make install_dev' to continue with installation.";

# Asset development installation
install_dev:
	@echo "Make sure your virtual environment is activated before running this command.";
	@echo "Correct parent directory. Proceeding with installation...";
	-pip install pokepy==0.4.0; \
	pip install -r src/requirements.txt; \
	pip install langchain streamlit; \
	echo "Installation complete. Run 'make install_test' to set up test environment.";

# Test environment setup
install_test:
	@echo "Make sure your virtual environment is activated before running this command."; \
	echo "Setting up test environment..."; \
	pip install -r src/test_requirements.txt; \
	echo "Test environment setup complete.";

# Run the API server
api_server:
	@echo "Make sure your virtual environment is activated before running this command.";
	@echo "Starting API server...";
	@export $$(grep -v '^#' .env | xargs) && \
	if [ -z "$$API_HOST" ] || [ -z "$$API_PORT" ]; then \
		echo "API_HOST or API_PORT environment variables not set. Please define them in .env file."; \
		exit 1; \
	else \
		uvicorn app:app --reload --host $$API_HOST --port $$API_PORT; \
	fi;

# Display the UI with Streamlit
display_ui:
	@echo "Make sure your virtual environment is activated before running this command.";
	@echo "Launching UI with Streamlit...";
	@export $$(grep -v '^#' .env | xargs) && \
	streamlit run pokedex_ui.py;
