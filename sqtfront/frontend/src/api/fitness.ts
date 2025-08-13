import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';
// Ensure video feed URLs use the same origin to avoid CORS issues
const VIDEO_FEED_BASE_URL = window.location.protocol + '//' + window.location.hostname + ':5000';

export interface Mode {
  id: string;
  name: string;
}

export interface View {
  id: string;
  name: string;
}

export const fitnessApi = {
  // Get available training modes
  getModes: async (): Promise<Mode[]> => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/modes`);
      return response.data.modes;
    } catch (error) {
      console.error('Error fetching modes:', error);
      return [];
    }
  },
  
  // Get available view types
  getViews: async (): Promise<View[]> => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/views`);
      return response.data.views;
    } catch (error) {
      console.error('Error fetching views:', error);
      return [];
    }
  },
  
  // Get video feed URL
  getVideoFeedUrl: async (exerciseType: string = 'squat', viewType: string = 'side'): Promise<string> => {
    try {
      // Use direct URL construction instead of API call to avoid CORS issues
      return `${VIDEO_FEED_BASE_URL}/video_feed?exercise=${exerciseType}&view=${viewType}`;
    } catch (error) {
      console.error('Error fetching video feed URL:', error);
      return '';
    }
  },
  
  // Start live analysis
  startLiveAnalysis: async (mode: string = 'Beginner', exerciseType: string = 'squat', viewType: string = 'side', useEnhanced: boolean = true): Promise<void> => {
    await axios.post(`${API_BASE_URL}/set_mode`, { mode, exerciseType, viewType, useEnhanced });
    await axios.get(`${API_BASE_URL}/start_camera?exercise=${exerciseType}&view=${viewType}`);
  },
  
  // Stop live analysis
  stopLiveAnalysis: async (): Promise<void> => {
    await axios.get(`${API_BASE_URL}/stop_camera`);
  },
  
  // Upload video for analysis
  uploadVideo: async (
    file: File, 
    mode: string = 'Beginner', 
    exerciseType: string = 'squat',
    viewType: string = 'side',
    useEnhanced: boolean = true
  ): Promise<{
    original: string;
    processed: string;
  }> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('mode', mode);
    formData.append('exerciseType', exerciseType);
    formData.append('viewType', viewType);
    formData.append('useEnhanced', useEnhanced.toString());
    
    const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    
    if (response.data.status === 'success') {
      const result = {
        original: `${API_BASE_URL}${response.data.original}`,
        processed: `${API_BASE_URL}${response.data.processed}`
      };
      console.log('API_BASE_URL:', API_BASE_URL);
      console.log('Response data:', response.data);
      console.log('Constructed URLs:', result);
      return result;
    } else {
      throw new Error(response.data.message || 'Video processing failed');
    }
  },
  
  // Mock function for feedback - you can implement real feedback API later
  getFeedback: async (exerciseType: string = 'squat'): Promise<{ feedback: string }> => {
    // In a real app, fetch this from your backend
    const feedbackMap: Record<string, string[]> = {
      squat: [
        "Keep your back straight",
        "Lower your hips more",
        "Keep knees aligned with toes",
        "Good depth, maintain form"
      ],
      plank: [
        "Keep your core engaged",
        "Maintain a straight line from head to heels",
        "Avoid dropping your hips",
        "Keep your neck neutral"
      ]
    };
    
    // Randomly select a feedback message for the demo
    const messages = feedbackMap[exerciseType] || feedbackMap.squat;
    const randomIndex = Math.floor(Math.random() * messages.length);
    
    return { feedback: messages[randomIndex] };
  }
};