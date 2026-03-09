#!/bin/bash

# Quick start script for development/testing
# For production, use Docker or deployment scripts

echo "🚀 Starting CKD Prediction System..."

# Check if in correct directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Start backend
echo "📦 Starting Backend Server..."
cd server

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt > /dev/null 2>&1

# Create logs directory
mkdir -p logs

# Start backend in background
echo "Starting Flask API on http://localhost:5000"
python app.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start frontend
echo "📦 Starting Frontend Server..."
cd client

if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

echo "Starting React app on http://localhost:5173"
npm run dev &
FRONTEND_PID=$!
cd ..

# Trap Ctrl+C to cleanup
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Stopped"
    exit 0
}

trap cleanup INT TERM

echo ""
echo "✅ System is running!"
echo "   Frontend: http://localhost:5173"
echo "   Backend:  http://localhost:5000"
echo "   API Docs: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Wait
wait
