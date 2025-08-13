document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultContainer = document.getElementById('resultContainer');
    const originalVideo = document.getElementById('originalVideo');
    const processedVideo = document.getElementById('processedVideo');
    const downloadBtn = document.getElementById('downloadBtn');
    
    // Update file name display
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            fileName.textContent = this.files[0].name;
        } else {
            fileName.textContent = 'No file chosen';
        }
    });
    
    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Check if file is selected
        if (fileInput.files.length === 0) {
            alert('Please select a video file');
            return;
        }
        
        // Show loading indicator
        loadingIndicator.classList.remove('hidden');
        resultContainer.classList.add('hidden');
        
        // Create form data
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('mode', document.querySelector('input[name="mode"]:checked').value);
        
        // Submit the form via AJAX
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Hide loading indicator
                loadingIndicator.classList.add('hidden');
                
                // Ensure correct video paths with cache busting
                const timestamp = new Date().getTime();
                originalVideo.src = data.original + '?t=' + timestamp;
                processedVideo.src = data.processed + '?t=' + timestamp;
                
                // Force video reload
                originalVideo.load();
                processedVideo.load();
                
                // Set download link
                downloadBtn.href = data.processed;
                downloadBtn.download = 'analyzed_' + fileInput.files[0].name;
                
                // Show results
                resultContainer.classList.remove('hidden');
                
                // Debug video loading
                console.log('Original video path:', data.original);
                console.log('Processed video path:', data.processed);
                
                // Reset form
                fileName.textContent = 'No file chosen';
                fileInput.value = '';
                
                // Log when videos are ready
                originalVideo.addEventListener('loadeddata', () => console.log('Original video loaded'));
                processedVideo.addEventListener('loadeddata', () => console.log('Processed video loaded'));
                processedVideo.addEventListener('error', (e) => console.error('Error loading processed video:', e));
            } else {
                alert('Error: ' + data.message);
                loadingIndicator.classList.add('hidden');
            }
        })
        .catch(error => {
            console.error('Error uploading video:', error);
            alert('Error uploading video. Please try again.');
            loadingIndicator.classList.add('hidden');
        });
    });
});