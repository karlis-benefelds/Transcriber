class TranscriptionApp {
    constructor() {
        this.currentJobId = null;
        this.statusCheckInterval = null;
        this.isPolling = false;
        this.initializeElements();
        this.bindEvents();
        this.setupVisibilityHandling();
    }

    setupVisibilityHandling() {
        // Pause polling when tab is not visible to reduce server load
        document.addEventListener('visibilitychange', () => {
            if (document.hidden && this.isPolling) {
                console.log('Tab hidden, pausing status checks');
                this.clearStatusChecking();
            } else if (!document.hidden && this.currentJobId && !this.statusCheckInterval) {
                console.log('Tab visible again, resuming status checks');
                this.startStatusChecking();
            }
        });
    }

    initializeElements() {
        // Containers
        this.formContainer = document.getElementById('workflow-container');
        this.progressContainer = document.getElementById('progress-state');
        this.resultsContainer = document.getElementById('success-state');
        this.errorContainer = document.getElementById('error-state');

        // Form elements
        this.form = document.getElementById('transcription-form');
        this.submitBtn = document.getElementById('submit-btn');
        this.curlInput = document.getElementById('curl_command');
        this.audioSourceRadios = document.querySelectorAll('input[name="audio_source"]');
        this.audioFile = document.getElementById('audio_file');
        this.audioUrl = document.getElementById('audio_url');
        this.drivePath = document.getElementById('drive_path');
        this.privacyMode = document.getElementById('privacy_mode');

        // Source groups
        this.uploadGroup = document.getElementById('upload-section');
        this.urlGroup = document.getElementById('url-section');
        this.driveGroup = document.getElementById('drive-section');

        // File input elements
        this.filePlaceholder = document.querySelector('.file-placeholder');
        this.browseBtn = document.querySelector('.file-input-area');

        // Progress elements
        this.progressText = document.getElementById('progress-text');
        this.statusMessage = document.getElementById('status-message');

        // Result elements
        this.downloadPdf = document.getElementById('download-pdf');
        this.downloadCsv = document.getElementById('download-csv');
        this.newTranscriptionBtn = document.getElementById('new-transcription');

        // Error elements
        this.errorMessage = document.getElementById('error-message');
        this.retryBtn = document.getElementById('retry-btn');
    }

    bindEvents() {
        // Form submission
        if (this.form) {
            this.form.addEventListener('submit', (e) => this.handleFormSubmit(e));
        }

        // Audio source radio buttons
        if (this.audioSourceRadios) {
            this.audioSourceRadios.forEach(radio => {
                radio.addEventListener('change', () => this.handleSourceChange());
            });
        }

        // File input
        if (this.audioFile) {
            this.audioFile.addEventListener('change', () => this.handleFileSelect());
        }
        if (this.browseBtn) {
            this.browseBtn.addEventListener('click', () => this.audioFile?.click());
        }

        // Action buttons
        if (this.newTranscriptionBtn) {
            this.newTranscriptionBtn.addEventListener('click', () => this.resetForm());
        }
        if (this.retryBtn) {
            this.retryBtn.addEventListener('click', () => this.resetForm());
        }

        // Initial source setup
        this.handleSourceChange();
    }

    handleSourceChange() {
        const selectedSource = document.querySelector('input[name="audio_source"]:checked').value;
        
        // Hide all groups
        this.uploadGroup.classList.add('hidden');
        this.urlGroup.classList.add('hidden');
        this.driveGroup.classList.add('hidden');

        // Show selected group
        switch (selectedSource) {
            case 'upload':
                this.uploadGroup.classList.remove('hidden');
                break;
            case 'url':
                this.urlGroup.classList.remove('hidden');
                break;
            case 'drive':
                this.driveGroup.classList.remove('hidden');
                break;
        }
    }

    handleFileSelect() {
        const file = this.audioFile.files[0];
        if (file) {
            this.filePlaceholder.textContent = file.name;
            this.filePlaceholder.classList.add('has-file');
        } else {
            this.filePlaceholder.textContent = 'Choose file (MP3, MP4, WAV, M4A, AAC, OGG)';
            this.filePlaceholder.classList.remove('has-file');
        }
    }

    validateForm() {
        // Check cURL command
        if (!this.curlInput.value.trim()) {
            this.showError('Please provide the Forum cURL command.');
            return false;
        }

        // Check audio source
        const selectedSource = document.querySelector('input[name="audio_source"]:checked').value;
        
        switch (selectedSource) {
            case 'upload':
                if (!this.audioFile.files[0]) {
                    this.showError('Please select an audio/video file.');
                    return false;
                }
                break;
            case 'url':
                if (!this.audioUrl.value.trim()) {
                    this.showError('Please provide a direct URL to the audio/video file.');
                    return false;
                }
                // Basic URL validation
                try {
                    new URL(this.audioUrl.value.trim());
                } catch {
                    this.showError('Please provide a valid URL.');
                    return false;
                }
                break;
            case 'drive':
                if (!this.drivePath.value.trim()) {
                    this.showError('Please provide the Google Drive path.');
                    return false;
                }
                break;
        }

        return true;
    }

    async handleFormSubmit(e) {
        e.preventDefault();

        // Prevent multiple simultaneous submissions
        if (this.currentJobId || this.isPolling) {
            console.warn('Transcription already in progress');
            return;
        }

        if (!this.validateForm()) {
            return;
        }

        this.submitBtn.disabled = true;
        this.submitBtn.textContent = 'Starting...';

        const formData = new FormData();
        formData.append('curl_command', this.curlInput.value.trim());
        formData.append('privacy_mode', this.privacyMode.value);

        const selectedSource = document.querySelector('input[name="audio_source"]:checked').value;
        
        switch (selectedSource) {
            case 'upload':
                formData.append('audio_file', this.audioFile.files[0]);
                break;
            case 'url':
                formData.append('audio_url', this.audioUrl.value.trim());
                break;
            case 'drive':
                formData.append('drive_path', this.drivePath.value.trim());
                break;
        }

        try {
            const response = await fetch('/api/transcribe', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to start transcription');
            }

            this.currentJobId = data.job_id;
            this.showProgress();
            this.startStatusChecking();

        } catch (error) {
            this.showError(error.message);
            this.resetSubmitButton();
        }
    }

    async startStatusChecking() {
        // Clear any existing interval to prevent multiple polling loops
        if (this.statusCheckInterval) {
            clearInterval(this.statusCheckInterval);
            this.statusCheckInterval = null;
        }

        this.isPolling = true;
        let attemptCount = 0;
        const maxAttempts = 600; // 30 minutes max (600 attempts * 3 seconds = 1800 seconds)
        
        this.statusCheckInterval = setInterval(async () => {
            attemptCount++;
            
            // Safety check: stop polling after max attempts
            if (attemptCount > maxAttempts) {
                console.warn('Max polling attempts reached, stopping status checks');
                clearInterval(this.statusCheckInterval);
                this.statusCheckInterval = null;
                this.showError('Transcription timeout - process took too long. Please try again.');
                return;
            }

            try {
                // Checking transcription status
                const response = await fetch(`/api/status/${this.currentJobId}`);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const status = await response.json();
                // Status response received

                this.updateProgress(status);

                if (status.status === 'completed') {
                    console.log('Transcription completed, stopping polling');
                    clearInterval(this.statusCheckInterval);
                    this.statusCheckInterval = null;
                    this.showResults(status);
                } else if (status.status === 'error') {
                    console.log('Transcription error, stopping polling');
                    clearInterval(this.statusCheckInterval);
                    this.statusCheckInterval = null;
                    this.showError(status.message || 'Transcription failed');
                }

            } catch (error) {
                console.error(`Status check failed (attempt ${attemptCount}):`, error);
                
                // If we get too many consecutive failures, stop polling
                if (attemptCount > 10 && attemptCount % 10 === 0) {
                    console.warn(`Multiple status check failures (${attemptCount} attempts), but continuing...`);
                }
            }
        }, 3000); // Increased to 3 seconds to reduce server load
    }

    updateProgress(status) {
        const message = status.message || 'Processing...';
        this.statusMessage.textContent = message;
        
        // Update step indicators
        const step = this.getStepFromMessage(message);
        if (step) {
            this.updateStepIndicators(step);
        }
    }

    getStepFromMessage(message) {
        if (message.includes('class information') || message.includes('Extract')) return 1;
        if (message.includes('Forum') || message.includes('Fetch')) return 2;
        if (message.includes('audio') || message.includes('Processing audio')) return 3;
        if (message.includes('Transcrib') || message.includes('transcrib')) return 4;
        if (message.includes('Generat') || message.includes('PDF') || message.includes('CSV')) return 5;
        return null;
    }

    updateStepIndicators(currentStep) {
        const steps = document.querySelectorAll('.step-indicator');
        steps.forEach((step, index) => {
            const stepNumber = index + 1;
            step.classList.remove('active', 'completed');
            
            if (stepNumber < currentStep) {
                step.classList.add('completed');
            } else if (stepNumber === currentStep) {
                step.classList.add('active');
            }
        });
    }

    showProgress() {
        this.formContainer.classList.add('hidden');
        this.resultsContainer.classList.add('hidden');
        this.errorContainer.classList.add('hidden');
        this.progressContainer.classList.remove('hidden');

        // Reset progress message and start with first step
        this.statusMessage.textContent = '🚀 Starting transcription process...';
        this.updateStepIndicators(1);
    }

    showResults(status) {
        // Ensure polling is stopped
        this.clearStatusChecking();
        
        // Mark all steps as completed before hiding progress
        this.updateStepIndicators(6); // Beyond the last step to mark all complete
        
        setTimeout(() => {
            this.progressContainer.classList.add('hidden');
            this.errorContainer.classList.add('hidden');
            this.resultsContainer.classList.remove('hidden');
        }, 1000); // Brief delay to show all steps completed

        // Set download links
        this.downloadPdf.href = `/api/download/${this.currentJobId}/pdf`;
        this.downloadCsv.href = `/api/download/${this.currentJobId}/csv`;

        // Update download button text based on privacy mode
        if (this.privacyMode.value === 'both') {
            this.downloadPdf.textContent = 'Download PDF (ZIP - Both Versions)';
            this.downloadCsv.textContent = 'Download CSV (ZIP - Both Versions)';
        } else if (this.privacyMode.value === 'ids') {
            this.downloadPdf.textContent = 'Download PDF (IDs)';
            this.downloadCsv.textContent = 'Download CSV (IDs)';
        } else {
            this.downloadPdf.textContent = 'Download PDF (Names)';
            this.downloadCsv.textContent = 'Download CSV (Names)';
        }
    }

    showError(message) {
        this.formContainer.classList.add('hidden');
        this.progressContainer.classList.add('hidden');
        this.resultsContainer.classList.add('hidden');
        this.errorContainer.classList.remove('hidden');

        this.errorMessage.textContent = message;

        // Always clear polling interval
        this.clearStatusChecking();
    }

    clearStatusChecking() {
        if (this.statusCheckInterval) {
            console.log('Clearing status check interval');
            clearInterval(this.statusCheckInterval);
            this.statusCheckInterval = null;
        }
        this.isPolling = false;
    }

    resetForm() {
        // Reset all containers
        this.formContainer.classList.remove('hidden');
        this.progressContainer.classList.add('hidden');
        this.resultsContainer.classList.add('hidden');
        this.errorContainer.classList.add('hidden');

        // Reset form
        this.form.reset();
        this.handleFileSelect(); // Reset file display
        this.handleSourceChange(); // Reset source groups
        this.resetSubmitButton();

        // Clear job tracking and polling
        this.currentJobId = null;
        this.clearStatusChecking();
    }

    resetSubmitButton() {
        this.submitBtn.disabled = false;
        this.submitBtn.textContent = 'Start Transcription';
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    // Only initialize if not already handled by inline scripts
    if (typeof window.transcriptionAppInitialized === 'undefined' || !window.transcriptionAppInitialized) {
        console.log('Initializing TranscriptionApp from script.js');
        window.transcriptionAppInitialized = true;
        new TranscriptionApp();
    } else {
        console.log('TranscriptionApp already initialized, skipping...');
    }
});