.PHONY: setup server agent-deploy clean

# Setup server environment
setup:
	python3 -m venv venv
	./venv/bin/pip install -r requirements/server.txt
	mkdir -p data/models data/scripts data/uploads
	@echo "Setup complete. Run 'make server' to start."

# Run server
server:
	./venv/bin/python -m server.main

# Deploy agent to Raspberry Pi (requires SSH access)
# Usage: make agent-deploy RPI_HOST=pi@192.168.1.100
agent-deploy:
	@if [ -z "$(RPI_HOST)" ]; then echo "Usage: make agent-deploy RPI_HOST=pi@192.168.1.100"; exit 1; fi
	scp -r agent/ $(RPI_HOST):~/edge-bench-agent/
	ssh $(RPI_HOST) "cd ~/edge-bench-agent && chmod +x install.sh && ./install.sh"

# Clean generated files
clean:
	rm -rf data/
	rm -rf venv/
	rm -rf __pycache__/
	find . -name "*.pyc" -delete

# =============================================================================
# ECCV 2026 Integration
# =============================================================================

eccv-models:
	@echo "Copying ECCV models to edge-bench..."
	mkdir -p models
	cp -v ../export/mobilenetv2_int8_ptq_*.tflite models/ 2>/dev/null || true
	cp -v ../export/mobilenetv1_int8_ptq_*.tflite models/ 2>/dev/null || true
	cp -v ../export/efficientnet_lite0_int8_ptq_*.tflite models/ 2>/dev/null || true
	cp -v ../export/efficientnet_int8_ptq_*.tflite models/ 2>/dev/null || true
	cp -v ../export/resnet50_int8_ptq_*.tflite models/ 2>/dev/null || true
	@echo "Models ready in: models/"
	@ls -la models/*.tflite 2>/dev/null || echo "No models found"

eccv-benchmark:
	@echo "Running ECCV benchmark via edge-bench..."
	cd .. && poetry run python scripts/9.9_run_edgebench.py \
		--server http://localhost:8000 \
		--runs 100 \
		--export-csv results/T4_edgetpu_final.csv

eccv-rpi-benchmark:
	@if [ -z "$(RPI_HOST)" ]; then echo "Usage: make eccv-rpi-benchmark RPI_HOST=pi@192.168.1.100"; exit 1; fi
	@echo "Deploying models to $(RPI_HOST)..."
	scp models/*.tflite "$(RPI_HOST):~/models/"
	scp scripts/benchmark_tflite.py "$(RPI_HOST):~/"
	scp scripts/benchmark_eccv_models.py "$(RPI_HOST):~/"
	@echo "Running benchmark on RPi..."
	ssh "$(RPI_HOST)" "cd ~ && python3 benchmark_eccv_models.py --local --models-dir models --output eccv_results.json --csv T4_edgetpu.csv"
	@echo "Downloading results..."
	scp "$(RPI_HOST):~/eccv_results.json" "../results/"
	scp "$(RPI_HOST):~/T4_edgetpu.csv" "../results/T4_edgetpu_final.csv"
	@echo "Done! Results in: ../results/"
