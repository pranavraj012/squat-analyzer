# Flask Squat Analysis AI

A real-time AI-powered exercise form analyzer built with Flask and React. This application uses computer vision to analyze squat form and provide instant feedback to help users improve their technique.

## Features

- Real-time squat form analysis using webcam  
- Video upload and analysis  
- Professional and Beginner modes  
- Instant visual feedback on form correction  
- Automatic rep counting  
- Exercise statistics tracking  
- Modern React frontend with Tailwind CSS  
- Responsive design for all devices  

## Technology Stack

### Backend

- Flask  
- OpenCV  
- Mediapipe  
- NumPy  

### Frontend

- React  
- TypeScript  
- Tailwind CSS  
- Framer Motion  
- Lucide Icons  

## Setup

### Prerequisites

- Python 3.8+  
- Node.js 16+  
- npm or yarn  

### Backend Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/flask-squat-analysis.git
   cd flask-squat-analysis
   ````

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask server:

   ```bash
   python app.py
   ```

### Frontend Setup

1. Navigate to the frontend directory:

   ```bash
   cd frontend
   ```

2. Install dependencies:

   ```bash
   npm install
   # or
   yarn install
   ```

3. Start the development server:

   ```bash
   npm run dev
   # or
   yarn dev
   ```

## Usage

1. Access the application at `http://localhost:5173`
2. Choose between **Live Analysis** or **Video Upload**
3. Select your experience level (Beginner/Professional)
4. For live analysis:

   * Allow camera access
   * Stand sideways to the camera
   * Maintain proper distance (full body should be visible)
   * Begin exercising
5. For video upload:

   * Upload a video file
   * Wait for processing
   * View results

