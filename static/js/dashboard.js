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
    
    // Initialize interactive elements
    initializeChartToggles();
    initializeDataTableSorting();
    setupExportButtons();
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
    const exportButtons = document.querySelectorAll('.export-button');
    exportButtons.forEach(button => {
        button.addEventListener('click', function() {
            const exportType = this.textContent.includes('CSV') ? 'csv' : 'excel';
            fetch(`/api/export/${exportType}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`Failed to export data as ${exportType.toUpperCase()}`);
                    }
                    return response.blob();
                })
                .then(blob => {
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `performance_data.${exportType}`;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                })
                .catch(error => {
                    alert(error.message);
                });
        });
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
