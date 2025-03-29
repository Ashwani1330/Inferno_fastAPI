// Dashboard event handlers and utilities

// Tab switching functionality
function openTab(evt, tabName) {
    // Hide all tab content
    const tabContents = document.getElementsByClassName("tab-content");
    for (let i = 0; i < tabContents.length; i++) {
        tabContents[i].style.display = "none";
    }

    // Remove "active" class from all tab buttons
    const tabButtons = document.getElementsByClassName("tab-button");
    for (let i = 0; i < tabButtons.length; i++) {
        tabButtons[i].className = tabButtons[i].className.replace(" active", "");
    }

    // Show the current tab and add "active" class to the button
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
    
    // Update URL with hash for direct tab access
    window.location.hash = tabName;
}

// Handle direct navigation to tabs via URL hash
document.addEventListener('DOMContentLoaded', function() {
    // Create overlay for expanded charts if it doesn't exist
    if (!document.querySelector('.chart-overlay')) {
        const overlay = document.createElement('div');
        overlay.className = 'chart-overlay';
        document.body.appendChild(overlay);
    }
    
    // Default to overview tab if no hash
    let activeTab = 'overview';
    
    // Check if hash exists and corresponds to a valid tab
    if (window.location.hash) {
        const hash = window.location.hash.substring(1);
        if (document.getElementById(hash)) {
            activeTab = hash;
        }
    }
    
    // Show the active tab content and add "active" class to the button
    document.getElementById(activeTab).style.display = "block";
    const tabButton = document.querySelector(`button[onclick="openTab(event, '${activeTab}')"]`);
    if (tabButton) {
        tabButton.className += " active";
    }
    
    // Debug log to check if scatterplot matrix is loaded
    const scatterplotImg = document.querySelector('#scatterplot-matrix');
    if (scatterplotImg) {
        console.log("Scatterplot Matrix found in DOM:", scatterplotImg.src.substring(0, 100) + "...");
        
        // Add event listeners for the scatterplot specifically
        scatterplotImg.addEventListener('load', function() {
            console.log("Scatterplot Matrix loaded successfully");
        });
        
        scatterplotImg.addEventListener('error', function() {
            console.error("Scatterplot Matrix failed to load");
        });
    } else {
        console.warn("Scatterplot Matrix element not found in DOM");
    }
    
    // Debug log for new visualizations
    const ptImg = document.querySelector('img[alt="Performance Trend Over Time"]');
    if (ptImg) {
        ptImg.addEventListener('load', function() {
            console.log("Performance Trend Over Time loaded successfully");
        });
        ptImg.addEventListener('error', function() {
            console.error("Error loading Performance Trend Over Time visualization");
        });
    }
    const taskSpeedImg = document.querySelector('img[alt="Task Completion Speed"]');
    if (taskSpeedImg) {
        taskSpeedImg.addEventListener('load', function() {
            console.log("Task Completion Speed visualization loaded successfully");
        });
        taskSpeedImg.addEventListener('error', function() {
            console.error("Error loading Task Completion Speed visualization");
        });
    }
    const featureImpactImg = document.querySelector('img[alt="Feature Impact"]');
    if (featureImpactImg) {
        featureImpactImg.addEventListener('load', function() {
            console.log("Feature Impact visualization loaded successfully");
        });
        featureImpactImg.addEventListener('error', function() {
            console.error("Error loading Feature Impact visualization");
        });
    }
    
    // Initialize interactive elements
    initializeChartToggles();
    initializeDataTableSorting();
    setupExportButtons();
    
    // Verify all chart images
    verifyChartImages();
    
    // Fix jQuery-like selector for older browsers
    if (!document.querySelector('.chart-container:has(#scatterplot-matrix)')) {
        // Polyfill the :has selector for browsers that don't support it
        const scatterplotTitle = Array.from(document.querySelectorAll('.chart-container h3')).find(
            h3 => h3.textContent.includes('Scatterplot Matrix')
        );
        if (scatterplotTitle) {
            window.scatterplotContainer = scatterplotTitle.closest('.chart-container');
        }
    }
});

// Toggle full size for charts when clicked
function initializeChartToggles() {
    const charts = document.querySelectorAll('.responsive-chart');
    const overlay = document.querySelector('.chart-overlay');
    
    charts.forEach(chart => {
        chart.addEventListener('click', function() {
            this.classList.toggle('expanded');
            
            if (this.classList.contains('expanded')) {
                overlay.classList.add('active');
                document.body.style.overflow = 'hidden'; // Prevent scrolling when viewing expanded chart
                
                // Add close button to expanded chart
                let closeBtn = document.createElement('button');
                closeBtn.innerHTML = '×';
                closeBtn.className = 'chart-close-btn';
                closeBtn.style.position = 'absolute';
                closeBtn.style.top = '10px';
                closeBtn.style.right = '10px';
                closeBtn.style.backgroundColor = '#e74c3c';
                closeBtn.style.color = 'white';
                closeBtn.style.border = 'none';
                closeBtn.style.borderRadius = '50%';
                closeBtn.style.width = '30px';
                closeBtn.style.height = '30px';
                closeBtn.style.fontSize = '20px';
                closeBtn.style.cursor = 'pointer';
                closeBtn.style.zIndex = '1001';
                
                this.parentNode.appendChild(closeBtn);
                
                // Add description to expanded view
                const description = this.nextElementSibling;
                if (description && description.classList.contains('chart-description')) {
                    const expandedDesc = description.cloneNode(true);
                    expandedDesc.style.position = 'absolute';
                    expandedDesc.style.bottom = '10px';
                    expandedDesc.style.left = '0';
                    expandedDesc.style.width = '100%';
                    expandedDesc.style.textAlign = 'center';
                    expandedDesc.style.background = 'rgba(255,255,255,0.8)';
                    expandedDesc.style.padding = '10px';
                    expandedDesc.style.borderRadius = '0 0 8px 8px';
                    expandedDesc.className = 'expanded-description';
                    this.parentNode.appendChild(expandedDesc);
                }
                
                closeBtn.addEventListener('click', function(e) {
                    e.stopPropagation();
                    closeExpandedChart(chart);
                });
            } else {
                closeExpandedChart(chart);
            }
        });
    });
    
    // Close expanded chart when clicking on overlay
    overlay.addEventListener('click', function() {
        const expandedChart = document.querySelector('.responsive-chart.expanded');
        if (expandedChart) {
            closeExpandedChart(expandedChart);
        }
    });
    
    // Close expanded chart with ESC key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            const expandedChart = document.querySelector('.responsive-chart.expanded');
            if (expandedChart) {
                closeExpandedChart(expandedChart);
            }
        }
    });
}

// Helper function to close expanded chart
function closeExpandedChart(chart) {
    chart.classList.remove('expanded');
    document.querySelector('.chart-overlay').classList.remove('active');
    document.body.style.overflow = 'auto'; // Re-enable scrolling
    
    // Remove close button if exists
    const closeBtn = chart.parentNode.querySelector('.chart-close-btn');
    if (closeBtn) {
        closeBtn.remove();
    }
    
    // Remove expanded description if exists
    const expandedDesc = chart.parentNode.querySelector('.expanded-description');
    if (expandedDesc) {
        expandedDesc.remove();
    }
}

// Initialize data table sorting functionality
function initializeDataTableSorting() {
    const tables = document.querySelectorAll('.data-table');
    
    tables.forEach(table => {
        const headers = table.querySelectorAll('th');
        
        headers.forEach((header, index) => {
            // Skip columns that shouldn't be sortable
            if (header.classList.contains('no-sort')) {
                return;
            }
            
            // Add sorting indicators
            header.innerHTML += '<span class="sort-indicator"> ↕</span>';
            header.style.cursor = 'pointer';
            header.style.userSelect = 'none';
            
            // Add click event for sorting
            header.addEventListener('click', function() {
                sortTable(table, index, this);
            });
        });
    });
}

// Sort table by column
function sortTable(table, column, header) {
    const rows = Array.from(table.querySelectorAll('tbody tr'));
    const direction = header.getAttribute('data-sort') === 'asc' ? 'desc' : 'asc';
    
    // Update all headers to show they're not the current sort column
    table.querySelectorAll('th').forEach(th => {
        th.querySelector('.sort-indicator').innerHTML = ' ↕';
        th.removeAttribute('data-sort');
    });
    
    // Update this header to show it's the current sort column
    header.setAttribute('data-sort', direction);
    header.querySelector('.sort-indicator').innerHTML = direction === 'asc' ? ' ↑' : ' ↓';
    
    // Sort the rows
    rows.sort((a, b) => {
        let valueA = a.cells[column].textContent.trim();
        let valueB = b.cells[column].textContent.trim();
        
        // Check if the values are numbers
        if (!isNaN(valueA) && !isNaN(valueB)) {
            return direction === 'asc' 
                ? parseFloat(valueA) - parseFloat(valueB)
                : parseFloat(valueB) - parseFloat(valueA);
        }
        
        // Check if values are dates in format YYYY-MM-DD
        if (/^\d{4}-\d{2}-\d{2}$/.test(valueA) && /^\d{4}-\d{2}-\d{2}$/.test(valueB)) {
            return direction === 'asc'
                ? new Date(valueA) - new Date(valueB)
                : new Date(valueB) - new Date(valueA);
        }
        
        // Otherwise treat as strings
        return direction === 'asc'
            ? valueA.localeCompare(valueB)
            : valueB.localeCompare(valueA);
    });
    
    // Re-add the sorted rows to the table
    const tbody = table.querySelector('tbody');
    rows.forEach(row => tbody.appendChild(row));
}

// Setup export buttons functionality
function setupExportButtons() {
    // Direct CSV export
    document.getElementById('csv-export').addEventListener('click', function() {
        window.location.href = '/api/export/csv';
    });
    
    // Direct Excel export
    document.getElementById('excel-export').addEventListener('click', function() {
        window.location.href = '/api/export/excel';
    });
}

// Format date for display
function formatDate(dateString) {
    const options = { year: 'numeric', month: 'short', day: 'numeric' };
    const date = new Date(dateString);
    return date.toLocaleDateString(undefined, options);
}

// Add a print function for research reports
function printDashboardContent() {
    const activeTab = document.querySelector('.tab-content[style*="block"]');
    if (!activeTab) return;
    
    const printWindow = window.open('', '_blank');
    
    // Generate the print content
    let content = `
        <html>
        <head>
            <title>Inferno VR Analytics - Research Report</title>
            <style>
                body { font-family: 'Segoe UI', Arial, sans-serif; color: #333; }
                h1, h2 { color: #2c3e50; }
                img { max-width: 100%; margin: 15px 0; }
                .header { text-align: center; margin-bottom: 30px; }
                .date { text-align: right; font-size: 0.9em; color: #777; }
                .report-section { margin-bottom: 30px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Inferno VR Fire-Safety Training</h1>
                <h2>Research Analytics Report</h2>
                <p class="date">Generated: ${new Date().toLocaleString()}</p>
            </div>
    `;
    
    // Add tab content
    content += `<div class="report-section">${activeTab.innerHTML}</div>`;
    
    // Close the HTML
    content += `
        </body>
        </html>
    `;
    
    // Write to the new window and print
    printWindow.document.open();
    printWindow.document.write(content);
    printWindow.document.close();
    
    // Wait for images to load before printing
    setTimeout(() => {
        printWindow.print();
    }, 500);
}

// Add improved error handling for chart images
function handleImageError(img) {
    console.error(`Failed to load image: ${img.alt}`);
    
    // Log additional details about the image
    console.error(`Image source length: ${img.src.length}`);
    console.error(`Image source starts with: ${img.src.substring(0, 30)}...`);
    
    // Replace the broken image with an error message
    const container = img.parentElement;
    const errorMsg = document.createElement('div');
    errorMsg.className = 'image-error';
    errorMsg.innerHTML = `
        <div class="alert alert-warning">
            <i class="fas fa-exclamation-triangle"></i> 
            Failed to load visualization. The data may be insufficient or the visualization generation timed out.
            <br><br>
            <button class="btn btn-sm btn-primary refresh-viz-btn">Regenerate Visualization</button>
        </div>
    `;
    container.replaceChild(errorMsg, img);
    
    // Add click handler for the regenerate button
    const refreshBtn = errorMsg.querySelector('.refresh-viz-btn');
    refreshBtn.addEventListener('click', function() {
        this.disabled = true;
        this.innerHTML = 'Regenerating...';
        
        // Call the refresh endpoint to regenerate the dashboard data
        fetch('/analytics/dashboard/refresh')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    // Reload the page to show the refreshed data
                    window.location.reload();
                } else {
                    errorMsg.innerHTML = `
                        <div class="alert alert-danger">
                            <i class="fas fa-exclamation-circle"></i> 
                            Failed to regenerate: ${data.message || 'Unknown error'}
                        </div>
                    `;
                }
            })
            .catch(error => {
                errorMsg.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-circle"></i> 
                        Error: ${error.message || 'Failed to contact server'}
                    </div>
                `;
            });
    });
}

// Function to regenerate alternative visualizations
function regenerateVisualizations() {
    const correlationsTab = document.getElementById('correlations');
    
    // Find all visualization containers in the correlations tab
    const vizContainers = correlationsTab.querySelectorAll('.chart-container:has(.image-error), .chart-container:has(.no-data)');
    
    vizContainers.forEach(container => {
        // Save the original title
        const title = container.querySelector('h3').innerHTML;
        
        // Replace with loading indicator
        container.innerHTML = `
            <h3>${title}</h3>
            <div class="generating-indicator">
                <div class="spinner-border text-primary" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="text-center mt-4">Generating visualizations... (This may take up to 30 seconds)</p>
            </div>
        `;
    });
    
    // Call the endpoint to regenerate the visualizations
    fetch('/analytics/dashboard/refresh')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Reload the page to show the regenerated data
                window.location.reload();
            } else {
                // Update containers with error message
                vizContainers.forEach(container => {
                    const title = container.querySelector('h3').innerHTML;
                    container.innerHTML = `
                        <h3>${title}</h3>
                        <div class="image-error">
                            <div class="alert alert-danger">
                                <i class="fas fa-exclamation-circle"></i> 
                                Failed to generate visualizations: ${data.message || 'Unknown error'}
                            </div>
                        </div>
                    `;
                });
            }
        })
        .catch(error => {
            // Update containers with error message
            vizContainers.forEach(container => {
                const title = container.querySelector('h3').innerHTML;
                container.innerHTML = `
                    <h3>${title}</h3>
                    <div class="image-error">
                        <div class="alert alert-danger">
                            <i class="fas fa-exclamation-circle"></i> 
                            Error: ${error.message || 'Failed to contact server'}
                        </div>
                    </div>
                `;
            });
        });
}

// Replace the old regenerateScatterplot function with the new regenerateVisualizations function
function regenerateScatterplot() {
    regenerateVisualizations();
}

// Add a function to check all images on load
function verifyChartImages() {
    const charts = document.querySelectorAll('.responsive-chart');
    charts.forEach(chart => {
        // Check if the image has a valid src
        if (chart.src) {
            // Check minimal source length for base64 images
            if (chart.src.startsWith('data:image') && chart.src.length < 100) {
                console.warn(`Suspiciously short image data for ${chart.alt}`);
                handleImageError(chart);
            }
            
            // Add error handler
            chart.addEventListener('error', function() {
                handleImageError(this);
            });
        } else {
            console.warn(`Missing source for chart: ${chart.alt}`);
            handleImageError(chart);
        }
    });
}
