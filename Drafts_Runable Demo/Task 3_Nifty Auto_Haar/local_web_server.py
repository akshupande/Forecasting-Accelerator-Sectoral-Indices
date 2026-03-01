
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
import os
import json
import pandas as pd
from datetime import datetime

class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = self.create_dashboard()
            self.wfile.write(html.encode('utf-8'))
            
        elif self.path == '/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            data = self.get_latest_data()
            self.wfile.write(json.dumps(data).encode('utf-8'))
            
        elif self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            status = self.get_mlops_status()
            self.wfile.write(json.dumps(status).encode('utf-8'))
            
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/run':
            import subprocess
            result = subprocess.run(['python', 'intl.py'], 
                                  capture_output=True, text=True)
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            response = {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr
            }
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def create_dashboard(self):
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>NIFTY AUTO Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }
                .card {
                    background: white;
                    padding: 20px;
                    margin: 10px 0;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .btn {
                    background: #4CAF50;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                }
                .btn:hover {
                    background: #45a049;
                }
                .grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 10px 0;
                }
                th, td {
                    padding: 10px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f2f2f2;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>NIFTY AUTO Forecasting Dashboard</h1>
                <p>Smart MLops Pipeline - Local Deployment</p>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h2>Quick Actions</h2>
                    <button class="btn" onclick="runPipeline()">Run Forecast Pipeline</button>
                    <button class="btn" onclick="refreshData()">Refresh Data</button>
                    <div id="status" style="margin-top: 10px;"></div>
                </div>
                
                <div class="card">
                    <h2>Latest Run Info</h2>
                    <div id="runInfo">Loading...</div>
                </div>
            </div>
            
            <div class="card">
                <h2>Latest Forecast</h2>
                <div style="height: 400px;">
                    <canvas id="forecastChart"></canvas>
                </div>
                <div id="forecastTable" style="margin-top: 20px;"></div>
            </div>
            
            <div class="card">
                <h2>MLops History</h2>
                <div style="height: 300px;">
                    <canvas id="historyChart"></canvas>
                </div>
            </div>
            
            <script>
                let forecastChart = null;
                let historyChart = null;
                
                // Load data on page load
                document.addEventListener('DOMContentLoaded', function() {
                    loadData();
                    loadHistory();
                });
                
                function loadData() {
                    fetch('/data')
                        .then(response => response.json())
                        .then(data => {
                            updateForecastChart(data);
                            updateRunInfo(data.runInfo);
                        });
                }
                
                function loadHistory() {
                    fetch('/status')
                        .then(response => response.json())
                        .then(data => {
                            updateHistoryChart(data);
                        });
                }
                
                function runPipeline() {
                    document.getElementById('status').innerHTML = 'Running...';
                    
                    fetch('/run', { method: 'POST' })
                        .then(response => response.json())
                        .then(result => {
                            if (result.success) {
                                document.getElementById('status').innerHTML = 'Success! Refreshing data...';
                                setTimeout(() => {
                                    loadData();
                                    loadHistory();
                                    document.getElementById('status').innerHTML = '';
                                }, 2000);
                            } else {
                                document.getElementById('status').innerHTML = 'Error: ' + result.error;
                            }
                        });
                }
                
                function refreshData() {
                    loadData();
                    loadHistory();
                    document.getElementById('status').innerHTML = 'Data refreshed!';
                    setTimeout(() => {
                        document.getElementById('status').innerHTML = '';
                    }, 2000);
                }
                
                function updateForecastChart(data) {
                    const ctx = document.getElementById('forecastChart').getContext('2d');
                    
                    if (forecastChart) {
                        forecastChart.destroy();
                    }
                    
                    forecastChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: data.dates,
                            datasets: [
                                {
                                    label: 'Actual Price',
                                    data: data.actual,
                                    borderColor: 'black',
                                    backgroundColor: 'rgba(0,0,0,0.1)',
                                    borderWidth: 3,
                                    fill: false
                                },
                                {
                                    label: 'Predicted Price',
                                    data: data.predicted,
                                    borderColor: 'blue',
                                    backgroundColor: 'rgba(0,0,255,0.1)',
                                    borderWidth: 2,
                                    borderDash: [5, 5],
                                    fill: false
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Forecast vs Actual'
                                }
                            },
                            scales: {
                                y: {
                                    beginAtZero: false,
                                    title: {
                                        display: true,
                                        text: 'Price'
                                    }
                                }
                            }
                        }
                    });
                    
                    // Update table
                    let tableHtml = '<table><tr><th>Date</th><th>Actual</th><th>Predicted</th><th>Error</th></tr>';
                    for (let i = 0; i < Math.min(10, data.dates.length); i++) {
                        const error = (data.actual[i] - data.predicted[i]).toFixed(2);
                        tableHtml += `<tr>
                            <td>${data.dates[i]}</td>
                            <td>${data.actual[i].toFixed(2)}</td>
                            <td>${data.predicted[i].toFixed(2)}</td>
                            <td>${error}</td>
                        </tr>`;
                    }
                    tableHtml += '</table>';
                    document.getElementById('forecastTable').innerHTML = tableHtml;
                }
                
                function updateRunInfo(runInfo) {
                    document.getElementById('runInfo').innerHTML = `
                        <p><strong>Run ID:</strong> ${runInfo.run_id}</p>
                        <p><strong>Best Model:</strong> ${runInfo.best_model}</p>
                        <p><strong>RMSE:</strong> ${runInfo.best_rmse.toFixed(2)}</p>
                        <p><strong>Test Period:</strong> ${runInfo.test_start} to ${runInfo.test_end}</p>
                    `;
                }
                
                function updateHistoryChart(data) {
                    const ctx = document.getElementById('historyChart').getContext('2d');
                    
                    if (historyChart) {
                        historyChart.destroy();
                    }
                    
                    historyChart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: data.run_ids,
                            datasets: [{
                                label: 'RMSE',
                                data: data.rmse_values,
                                borderColor: 'red',
                                backgroundColor: 'rgba(255,0,0,0.1)',
                                borderWidth: 2,
                                fill: true
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                title: {
                                    display: true,
                                    text: 'Model Performance History'
                                }
                            },
                            scales: {
                                y: {
                                    beginAtZero: false,
                                    title: {
                                        display: true,
                                        text: 'RMSE'
                                    }
                                }
                            }
                        }
                    });
                }
            </script>
        </body>
        </html>
        """
    
    def get_latest_data(self):
        try:
            if os.path.exists('latest_predictions.csv'):
                df = pd.read_csv('latest_predictions.csv')
                pred_cols = [col for col in df.columns if col.startswith('Pred_')]
                pred_col = pred_cols[0] if pred_cols else None
                
                run_info = self.get_run_info()
                
                return {
                    'dates': df['Date'].tail(20).tolist(),
                    'actual': df['Actual'].tail(20).tolist() if 'Actual' in df.columns else [],
                    'predicted': df[pred_col].tail(20).tolist() if pred_col else [],
                    'runInfo': run_info
                }
        except:
            pass
        return {'dates': [], 'actual': [], 'predicted': [], 'runInfo': {}}
    
    def get_run_info(self):
        try:
            if os.path.exists('smart_mlops_log.csv'):
                df = pd.read_csv('smart_mlops_log.csv')
                latest = df.iloc[-1]
                return {
                    'run_id': latest['run_id'],
                    'best_model': latest['best_model'],
                    'best_rmse': float(latest['best_rmse']),
                    'test_start': latest['test_start'],
                    'test_end': latest['test_end']
                }
        except:
            pass
        return {}
    
    def get_mlops_status(self):
        try:
            if os.path.exists('smart_mlops_log.csv'):
                df = pd.read_csv('smart_mlops_log.csv')
                return {
                    'run_ids': df['run_id'].tolist(),
                    'rmse_values': df['best_rmse'].tolist(),
                    'models': df['best_model'].tolist()
                }
        except:
            pass
        return {'run_ids': [], 'rmse_values': [], 'models': []}

def start_server():
    port = 8080
    server_address = ('', port)
    
    # Get local IP
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("=" * 60)
    print("NIFTY AUTO Local Web Server")
    print("=" * 60)
    print(f"Local access:  http://localhost:{port}")
    print(f"Network access: http://{local_ip}:{port}")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    # Open browser
    webbrowser.open(f'http://localhost:{port}')
    
    # Start server
    httpd = HTTPServer(server_address, DashboardHandler)
    httpd.serve_forever()

if __name__ == "__main__":
    start_server()