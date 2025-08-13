import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowLeft, Camera, Upload, BarChart3 } from 'lucide-react';

interface ExerciseCardProps {
  title: string;
  description: string;
  image: string;
  slug: string;
}

const ExerciseCard: React.FC<ExerciseCardProps> = ({ title, description, image, slug }) => {
  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      className="bg-gray-800/50 backdrop-blur-sm rounded-xl overflow-hidden border border-gray-700 flex flex-col"
    >
      <div className="h-56 overflow-hidden relative">
        <img 
          src={image} 
          alt={title} 
          className="w-full h-full object-cover"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-gray-900 to-transparent opacity-70"></div>
        <h3 className="absolute bottom-4 left-4 text-2xl font-bold">{title}</h3>
      </div>
      <div className="p-6 flex-grow">
        <p className="text-gray-300 mb-6">{description}</p>
        <div className="flex flex-col sm:flex-row gap-3">
          <Link 
            to={`/exercises/${slug}/live`}
            className="flex-1 bg-teal-600 hover:bg-teal-700 text-white py-2 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors"
          >
            <Camera size={18} />
            <span>Live Camera</span>
          </Link>
          <Link 
            to={`/exercises/${slug}/upload`}
            className="flex-1 bg-gray-700 hover:bg-gray-600 text-white py-2 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors"
          >
            <Upload size={18} />
            <span>Upload Video</span>
          </Link>
        </div>
      </div>
    </motion.div>
  );
};

const ExercisesPage: React.FC = () => {
  const exercises = [
    {
      title: "Squat Analysis",
      description: "Perfect your squat form with AI feedback on hip angle, knee position, and depth. Ideal for beginners and advanced athletes.",
      image: "http://localhost:5000/static/img/squat.jpg", 
      slug: "squat"
    },
    {
      title: "Plank Form",
      description: "Maintain proper plank position with real-time posture correction. Our AI detects spine alignment and body position.",
      image: "http://localhost:5000/static/img/plank.jpg",
      slug: "plank" 
    }
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5 }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-teal-900 text-white">
      <div className="container mx-auto px-6 py-12 max-w-6xl">
        <Link 
          to="/"
          className="flex items-center text-teal-400 hover:text-teal-300 transition-colors mb-10"
        >
          <ArrowLeft size={20} className="mr-2" />
          <span>Back to Home</span>
        </Link>
        
        <div className="flex items-center justify-between mb-12">
          <h1 className="text-3xl md:text-4xl font-bold">Exercise Selection</h1>
          <div className="hidden md:flex items-center bg-gray-800/70 backdrop-blur-sm px-4 py-2 rounded-lg">
            <BarChart3 className="text-teal-400 mr-2" size={20} />
            <span className="text-gray-300">Choose an exercise to analyze</span>
          </div>
        </div>
        
        <motion.div 
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="grid md:grid-cols-2 gap-8"
        >
          {exercises.map((exercise) => (
            <motion.div key={exercise.slug} variants={itemVariants}>
              <ExerciseCard {...exercise} />
            </motion.div>
          ))}
        </motion.div>
        
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.5 }}
          className="mt-16 text-center bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-8"
        >
          <h2 className="text-2xl font-bold mb-4">More exercises coming soon!</h2>
          <p className="text-gray-300 max-w-2xl mx-auto">
            Our team is working on adding more exercise analyses including pushups, lunges, and more.
            Stay tuned for updates!
          </p>
        </motion.div>
      </div>
    </div>
  );
};

export default ExercisesPage;
