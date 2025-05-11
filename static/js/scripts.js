// Handle file upload with AJAX
document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const statusDiv = document.getElementById('uploadStatus');
    
    statusDiv.innerHTML = '<div class="spinner-border text-primary" role="status"></div> Uploading...';
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            statusDiv.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
        } else {
            statusDiv.innerHTML = `<div class="alert alert-success">${data.success}</div>`;
            // Refresh the page to update stats
            setTimeout(() => location.reload(), 1500);
        }
    })
    .catch(error => {
        statusDiv.innerHTML = `<div class="alert alert-danger">Upload failed: ${error}</div>`;
    });
});

// Initialize county select with Select2
$(document).ready(function() {
    $('select[name="county_filter[]"]').select2({
        placeholder: "Select counties...",
        allowClear: true
    });
});