import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import LandingPage from './components/LandingPage';
import ExercisesPage from './components/ExercisesPage.tsx';
import TrainingInterface from './components/TrainingInterface';

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/exercises" element={<ExercisesPage />} />
        <Route path="/exercises/squat/live" element={<TrainingInterface exerciseType="squat" mode="live" onBack={() => window.history.back()} />} />
        <Route path="/exercises/squat/upload" element={<TrainingInterface exerciseType="squat" mode="upload" onBack={() => window.history.back()} />} />
        <Route path="/exercises/plank/live" element={<TrainingInterface exerciseType="plank" mode="live" onBack={() => window.history.back()} />} />
        <Route path="/exercises/plank/upload" element={<TrainingInterface exerciseType="plank" mode="upload" onBack={() => window.history.back()} />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
};

export default App;