import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion, useAnimation } from 'framer-motion';
import { ArrowRight, Dumbbell, Shield, Award, Activity } from 'lucide-react';

const LandingPage: React.FC = () => {
  const controls = useAnimation();

  useEffect(() => {
    controls.start('visible');
  }, [controls]);

  const containerVariants = {
    hidden: {},
    visible: {
      transition: {
        staggerChildren: 0.2
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.5 } }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-teal-900 text-white">
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8 }}
        className="container mx-auto px-6 pt-24 pb-12 max-w-6xl"
      >
        <div className="flex flex-col lg:flex-row items-center">
          <div className="lg:w-1/2 mb-12 lg:mb-0">
            <motion.h1 
              initial={{ opacity: 0, y: -30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.8 }}
              className="text-5xl md:text-6xl font-bold mb-6"
            >
              AI-Powered <span className="text-teal-400">Fitness</span> Analysis
            </motion.h1>
            <motion.p 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4, duration: 0.8 }}
              className="text-xl text-gray-300 mb-8"
            >
              Perfect your form, prevent injuries, and maximize workout efficiency with real-time AI exercise analysis.
            </motion.p>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6, duration: 0.8 }}
            >
              <Link 
                to="/exercises" 
                className="bg-teal-500 hover:bg-teal-600 text-white font-bold py-4 px-8 rounded-lg mr-4 transition-all duration-300 inline-flex items-center"
              >
                Get Started
                <ArrowRight className="ml-2" />
              </Link>
            </motion.div>
          </div>
          <div className="lg:w-1/2">
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.5, duration: 0.8 }}
              className="relative"
            >
              <div className="bg-teal-500/10 backdrop-blur-sm border border-teal-500/20 rounded-2xl overflow-hidden shadow-2xl">
                <video 
                  className="w-full rounded-lg"
                  autoPlay
                  loop
                  muted
                  playsInline
                  poster="http://localhost:5000/static/img/video-thumbnail.jpg"
                >
                  <source src="http://localhost:5000/static/outputs/output_sample.mp4" type="video/mp4" />
                  Your browser does not support the video tag.
                </video>
              </div>
              <div className="absolute -bottom-6 -right-6 h-24 w-24 bg-teal-400 rounded-full opacity-20 blur-lg"></div>
              <div className="absolute -top-6 -left-6 h-24 w-24 bg-teal-400 rounded-full opacity-20 blur-lg"></div>
            </motion.div>
          </div>
        </div>
      </motion.div>

      {/* Features Section */}
      <div className="py-16 bg-gray-900/50 backdrop-blur-sm">
        <div className="container mx-auto px-6 max-w-6xl">
          <motion.h2 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-3xl font-bold text-center mb-12"
          >
            Why Choose Our AI Fitness Trainer
          </motion.h2>
          
          <motion.div 
            variants={containerVariants}
            initial="hidden"
            animate={controls}
            className="grid md:grid-cols-3 gap-8"
          >
            <motion.div variants={itemVariants} className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-8 border border-gray-700 hover:border-teal-400/50 transition-all duration-300">
              <div className="bg-teal-500/20 rounded-full w-14 h-14 flex items-center justify-center mb-6">
                <Activity className="text-teal-400 h-7 w-7" />
              </div>
              <h3 className="text-xl font-semibold mb-4">Real-time Analysis</h3>
              <p className="text-gray-400">Get instant feedback on your exercise form as you workout, with AI-powered motion tracking.</p>
            </motion.div>
            
            <motion.div variants={itemVariants} className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-8 border border-gray-700 hover:border-teal-400/50 transition-all duration-300">
              <div className="bg-teal-500/20 rounded-full w-14 h-14 flex items-center justify-center mb-6">
                <Shield className="text-teal-400 h-7 w-7" />
              </div>
              <h3 className="text-xl font-semibold mb-4">Injury Prevention</h3>
              <p className="text-gray-400">Reduce risk of injuries with advanced posture detection algorithms that ensure proper alignment.</p>
            </motion.div>
            
            <motion.div variants={itemVariants} className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-8 border border-gray-700 hover:border-teal-400/50 transition-all duration-300">
              <div className="bg-teal-500/20 rounded-full w-14 h-14 flex items-center justify-center mb-6">
                <Award className="text-teal-400 h-7 w-7" />
              </div>
              <h3 className="text-xl font-semibold mb-4">Performance Tracking</h3>
              <p className="text-gray-400">Track your progress over time and see improvements in your form and technique.</p>
            </motion.div>
          </motion.div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="py-16 container mx-auto px-6 max-w-4xl">
        <motion.div 
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.8 }}
          className="bg-gradient-to-r from-teal-600/20 to-teal-400/20 backdrop-blur-sm rounded-2xl p-10 border border-teal-500/20 text-center"
        >
          <div className="inline-block bg-teal-500/20 p-3 rounded-full mb-6">
            <Dumbbell className="h-8 w-8 text-teal-400" />
          </div>
          <h2 className="text-3xl font-bold mb-4">Ready to improve your workout?</h2>
          <p className="text-xl text-gray-300 mb-6 max-w-xl mx-auto">
            Choose from our selection of AI-analyzed exercises and start perfecting your form today.
          </p>
          <Link 
            to="/exercises" 
            className="bg-teal-500 hover:bg-teal-600 text-white font-bold py-3 px-8 rounded-lg transition-all duration-300 inline-flex items-center"
          >
            Browse Exercises
            <ArrowRight className="ml-2" />
          </Link>
        </motion.div>
      </div>
      
      {/* Footer */}
      <footer className="py-10 bg-gray-900/80">
        <div className="container mx-auto px-6">
          <div className="text-center text-gray-400">
            <p>&copy; {new Date().getFullYear()} AI Fitness Trainer. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;