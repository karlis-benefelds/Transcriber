class AIChat {
    constructor() {
        this.uploadedFiles = [];
        this.initializeElements();
        this.bindEvents();
    }

    initializeElements() {
        // Modal elements
        this.aiModal = document.getElementById('ai-modal');
        this.aiAnalysisBtn = document.getElementById('ai-analysis-btn');
        this.closeModalBtn = document.getElementById('close-ai-modal');
        
        // Upload elements
        this.uploadArea = document.getElementById('upload-area');
        this.transcriptFiles = document.getElementById('transcript-files');
        this.addSourcesBtn = document.getElementById('add-sources-btn');
        
        // Source elements
        this.selectAllSources = document.getElementById('select-all-sources');
        this.sourcesList = document.getElementById('sources-list');
        
        // Chat elements
        this.chatMessages = document.getElementById('chat-messages');
        this.chatInput = document.getElementById('chat-input');
        this.sendBtn = document.getElementById('send-btn');
        this.initialAnalysisBtn = document.getElementById('initial-analysis-btn');
        this.exportPdfBtn = document.getElementById('export-pdf-btn');
        this.clearChatBtn = document.getElementById('clear-chat-btn');
    }

    bindEvents() {
        // Modal controls
        this.aiAnalysisBtn.addEventListener('click', () => this.openModal());
        this.closeModalBtn.addEventListener('click', () => this.closeModal());
        this.aiModal.addEventListener('click', (e) => {
            if (e.target === this.aiModal) this.closeModal();
        });
        
        // Escape key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !this.aiModal.classList.contains('hidden')) {
                this.closeModal();
            }
        });

        // File upload
        this.uploadArea.addEventListener('click', () => this.transcriptFiles.click());
        this.addSourcesBtn.addEventListener('click', () => this.transcriptFiles.click());
        this.transcriptFiles.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Source selection
        this.selectAllSources.addEventListener('change', (e) => this.handleSelectAll(e));
        
        // Drag and drop
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });
        
        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('dragover');
        });
        
        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            this.handleFileSelect(e);
        });

        // Chat controls
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.initialAnalysisBtn.addEventListener('click', () => this.generateInitialAnalysis());
        this.exportPdfBtn.addEventListener('click', () => this.exportToPDF());
        this.clearChatBtn.addEventListener('click', () => this.clearChat());
        
        // Enter to send (Shift+Enter for new line)
        this.chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!this.sendBtn.disabled) {
                    this.sendMessage();
                }
            }
        });
        
        // Enable/disable send button based on input
        this.chatInput.addEventListener('input', () => {
            this.updateSendButton();
        });
    }

    openModal() {
        this.aiModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }

    closeModal() {
        this.aiModal.classList.add('hidden');
        document.body.style.overflow = '';
    }

    handleFileSelect(e) {
        const files = e.type === 'drop' ? e.dataTransfer.files : e.target.files;
        
        for (let file of files) {
            if (this.isValidFile(file)) {
                this.addFile(file);
            } else {
                this.showError(`Invalid file type: ${file.name}. Only PDF and CSV files are supported.`);
            }
        }
        
        // Clear the input so the same file can be selected again
        this.transcriptFiles.value = '';
    }

    isValidFile(file) {
        const allowedTypes = ['application/pdf', 'text/csv', 'application/vnd.ms-excel'];
        const allowedExtensions = ['.pdf', '.csv'];
        
        return allowedTypes.includes(file.type) || 
               allowedExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
    }

    addFile(file) {
        // Check if file already exists
        if (this.uploadedFiles.some(f => f.name === file.name && f.size === file.size)) {
            this.showError(`File "${file.name}" is already uploaded.`);
            return;
        }
        
        this.uploadedFiles.push({
            file: file,
            name: file.name,
            selected: true
        });
        this.renderSourcesList();
        this.updateUI();
    }

    removeFile(index) {
        this.uploadedFiles.splice(index, 1);
        this.renderSourcesList();
        this.updateUI();
    }

    handleSelectAll(e) {
        const selectAll = e.target.checked;
        this.uploadedFiles.forEach(file => {
            file.selected = selectAll;
        });
        this.renderSourcesList();
        this.updateUI();
    }

    handleSourceToggle(index, selected) {
        this.uploadedFiles[index].selected = selected;
        this.updateSelectAllState();
        this.updateUI();
    }

    updateSelectAllState() {
        if (this.uploadedFiles.length === 0) {
            this.selectAllSources.checked = false;
            this.selectAllSources.indeterminate = false;
        } else {
            const selectedCount = this.uploadedFiles.filter(f => f.selected).length;
            if (selectedCount === 0) {
                this.selectAllSources.checked = false;
                this.selectAllSources.indeterminate = false;
            } else if (selectedCount === this.uploadedFiles.length) {
                this.selectAllSources.checked = true;
                this.selectAllSources.indeterminate = false;
            } else {
                this.selectAllSources.checked = false;
                this.selectAllSources.indeterminate = true;
            }
        }
    }

    renderSourcesList() {
        this.sourcesList.innerHTML = '';
        
        this.uploadedFiles.forEach((fileObj, index) => {
            const sourceItem = document.createElement('label');
            sourceItem.className = 'source-file';
            sourceItem.innerHTML = `
                <input type="checkbox" ${fileObj.selected ? 'checked' : ''} data-index="${index}">
                ${fileObj.name}
            `;
            
            const checkbox = sourceItem.querySelector('input[type="checkbox"]');
            checkbox.addEventListener('change', (e) => {
                this.handleSourceToggle(index, e.target.checked);
            });
            
            this.sourcesList.appendChild(sourceItem);
        });
        
        this.updateSelectAllState();
    }

    updateUI() {
        this.updateSendButton();
        this.initialAnalysisBtn.disabled = this.uploadedFiles.filter(f => f.selected).length === 0;
        this.updateExportButton();
    }
    
    updateExportButton() {
        // Enable export button if there are any chat messages beyond the welcome message
        const messages = this.chatMessages.querySelectorAll('.ai-message, .user-message');
        const hasActualConversation = messages.length > 1; // More than just the welcome message
        this.exportPdfBtn.disabled = !hasActualConversation;
    }

    updateSendButton() {
        const hasText = this.chatInput.value.trim().length > 0;
        const hasSelectedSources = this.uploadedFiles.some(f => f.selected);
        this.sendBtn.disabled = !hasText || !hasSelectedSources;
    }


    async sendMessage() {
        const message = this.chatInput.value.trim();
        const selectedFiles = this.uploadedFiles.filter(f => f.selected);
        if (!message || selectedFiles.length === 0) return;

        // Add user message
        this.addMessage(message, 'user');
        this.chatInput.value = '';
        this.updateSendButton();

        // Show typing indicator
        this.showTypingIndicator();

        try {
            const response = await this.callAI(message, false);
            this.hideTypingIndicator();
            this.addMessage(response, 'ai');
        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage('Sorry, I encountered an error processing your request. Please try again.', 'ai');
            console.error('AI API Error:', error);
        }
    }

    async generateInitialAnalysis() {
        const selectedFiles = this.uploadedFiles.filter(f => f.selected);
        if (selectedFiles.length === 0) return;

        this.initialAnalysisBtn.disabled = true;
        this.showTypingIndicator();

        try {
            const response = await this.callAI('', true);
            this.hideTypingIndicator();
            this.addMessage(response, 'ai');
        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage('Sorry, I encountered an error generating the initial analysis. Please try again.', 'ai');
            console.error('AI API Error:', error);
        } finally {
            this.initialAnalysisBtn.disabled = this.uploadedFiles.filter(f => f.selected).length === 0;
        }
    }

    async callAI(message, isInitialAnalysis) {
        const formData = new FormData();
        
        // Add only selected files
        const selectedFiles = this.uploadedFiles.filter(f => f.selected);
        selectedFiles.forEach((fileObj, index) => {
            formData.append(`file_${index}`, fileObj.file);
        });
        
        formData.append('message', message);
        formData.append('is_initial_analysis', isInitialAnalysis.toString());

        const response = await fetch('/api/ai-chat', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'AI service unavailable');
        }

        const data = await response.json();
        return data.response;
    }

    parseMarkdown(text) {
        // Simple markdown parser for basic formatting
        let html = text;
        
        // Convert headers
        html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        
        // Convert bold text
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        
        // Convert bullet points to list items
        html = html.replace(/^- (.+)$/gm, '|||LIST_ITEM|||$1|||END_LIST_ITEM|||');
        
        // Convert consecutive list items to proper lists
        html = html.replace(/((?:\|\|\|LIST_ITEM\|\|\|.*?\|\|\|END_LIST_ITEM\|\|\|\n?)+)/g, (match) => {
            const listItems = match.split('|||END_LIST_ITEM|||')
                .filter(item => item.trim())
                .map(item => item.replace('|||LIST_ITEM|||', '').trim())
                .map(item => `<li>${item}</li>`)
                .join('');
            return `<ul>${listItems}</ul>`;
        });
        
        // Clean up any remaining markers
        html = html.replace(/\|\|\|LIST_ITEM\|\|\|||\|\|\|END_LIST_ITEM\|\|\|/g, '');
        
        // Convert line breaks
        html = html.replace(/\n\n/g, '<br><br>');
        html = html.replace(/\n/g, '<br>');
        
        return html;
    }

    addMessage(content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `${type}-message`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        if (type === 'ai') {
            const formattedContent = this.parseMarkdown(content);
            messageContent.innerHTML = `<strong>AI Assistant:</strong><br><br>${formattedContent}`;
        } else {
            messageContent.textContent = content;
        }
        
        messageDiv.appendChild(messageContent);
        this.chatMessages.appendChild(messageDiv);
        
        // Update export button state
        this.updateExportButton();
        
        // Scroll to bottom
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'ai-message typing-indicator';
        typingDiv.id = 'typing-indicator';
        typingDiv.innerHTML = `
            <div class="message-content">
                <strong>AI Assistant:</strong>
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        
        this.chatMessages.appendChild(typingDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    clearChat() {
        // Keep only the initial welcome message
        const welcomeMessage = this.chatMessages.querySelector('.ai-message');
        this.chatMessages.innerHTML = '';
        if (welcomeMessage) {
            this.chatMessages.appendChild(welcomeMessage);
        }
        this.updateExportButton();
    }

    exportToPDF() {
        try {
            // Get jsPDF from the global window object
            const { jsPDF } = window.jspdf;
            const doc = new jsPDF();
            
            // Set up the document
            const pageWidth = doc.internal.pageSize.getWidth();
            const pageHeight = doc.internal.pageSize.getHeight();
            const margin = 20;
            const maxWidth = pageWidth - (margin * 2);
            
            // Add title
            doc.setFontSize(16);
            doc.setFont(undefined, 'bold');
            doc.text('AI Transcript Analysis Chat Export', margin, margin);
            
            // Add export date
            doc.setFontSize(10);
            doc.setFont(undefined, 'normal');
            const exportDate = new Date().toLocaleString();
            doc.text(`Exported: ${exportDate}`, margin, margin + 15);
            
            let yPosition = margin + 35;
            
            // Get all messages except typing indicators
            const messages = this.chatMessages.querySelectorAll('.ai-message:not(.typing-indicator), .user-message');
            
            messages.forEach((messageDiv, index) => {
                const isAI = messageDiv.classList.contains('ai-message');
                const messageContent = messageDiv.querySelector('.message-content');
                
                if (!messageContent) return;
                
                // Extract text content and clean it up
                let text = messageContent.textContent || messageContent.innerText;
                text = text.replace(/AI Assistant:\s*/, ''); // Remove AI Assistant prefix
                text = text.trim();
                
                // Skip empty messages
                if (!text) return;
                
                // Add message header
                doc.setFontSize(12);
                doc.setFont(undefined, 'bold');
                const sender = isAI ? 'AI Assistant' : 'User';
                
                // Check if we need a new page
                if (yPosition + 20 > pageHeight - margin) {
                    doc.addPage();
                    yPosition = margin;
                }
                
                doc.text(`${sender}:`, margin, yPosition);
                yPosition += 15;
                
                // Add message content
                doc.setFontSize(10);
                doc.setFont(undefined, 'normal');
                
                // Split text into lines that fit the page width
                const lines = doc.splitTextToSize(text, maxWidth);
                
                lines.forEach(line => {
                    // Check if we need a new page
                    if (yPosition + 5 > pageHeight - margin) {
                        doc.addPage();
                        yPosition = margin;
                    }
                    
                    doc.text(line, margin, yPosition);
                    yPosition += 5;
                });
                
                yPosition += 10; // Add space between messages
            });
            
            // Generate filename with timestamp
            const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
            const filename = `chat-export-${timestamp}.pdf`;
            
            // Save the PDF
            doc.save(filename);
            
        } catch (error) {
            console.error('Error exporting PDF:', error);
            this.showError('Failed to export PDF. Please try again.');
        }
    }

    showError(message) {
        // Create a temporary error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'ai-message';
        errorDiv.innerHTML = `
            <div class="message-content" style="background: #fee; border-color: #e74c3c; color: #c0392b;">
                <strong>Error:</strong> ${message}
            </div>
        `;
        
        this.chatMessages.appendChild(errorDiv);
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        
        // Remove error after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.remove();
            }
        }, 5000);
    }
}

// Initialize AI Chat when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AIChat();
});