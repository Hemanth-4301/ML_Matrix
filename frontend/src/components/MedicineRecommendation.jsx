import React, { useState } from "react";
import axios from "axios";
import "../App.css";
import { motion } from "framer-motion";
import heart from "../images/heart.png";
import Footer from "./Footer";

const MedicineRecommendation = () => {
  const [load, setLoad] = useState(false);
  const [symptom, setSymptom] = useState("");
  const [recommendation, setRecommendation] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoad(true);
    try {
      const response = await axios.post(
        "http://localhost:5000/api/symptoms/recommend",
        { symptom }
      );
      setTimeout(() => {
        setLoad(false);
      }, 500);

      setRecommendation(response.data);
      setError(null);
    } catch (err) {
      setError(err.response.data.message);
      setRecommendation(null);
      setLoad(false);
    }
  };

  return (
    <>
      <div
        id="home"
        className="flex flex-wrap justify-center items-center p-6 bg-yellow-200 "
      >
        <div className="w-full md:w-1/2 flex justify-center ">
          <motion.img
            initial={{ scale: 0.8, opacity: 8 }}
            whileInView={{ scale: 1, opacity: 1 }}
            transition={{ duration: 1 }}
            src={heart}
            alt="Heart illustration"
            height={600}
            width={600}
            className="flex flex-wrap heart"
          />
        </div>
        <div className="w-full md:w-1/2 text-center md:text-left px-4 md:px-8 lg:px-12">
          <h1 className="text-2xl md:text-4xl lg:text-5xl font-bold text-gray-800 mb-4">
            AI Medicine Recommender
          </h1>
          <p className="text-base md:text-lg text-gray-800 mb-6">
            Discover the right medicine for your needs with personalized
            recommendations. Empower your health decisions with our intelligent
            insights.
          </p>
          <a
            href="#finder"
            className="px-4 py-3 bg-black text-white font-semibold rounded-lg hover:bg-transparent hover:text-black border border-black transition-all duration-300 ease-in-out"
          >
            Get started
          </a>
        </div>
      </div>

      <section id="finder">
        <div className="flex justify-center items-center mt-0 px-4 py-20 bg-green-200">
          <motion.form
            initial={{ scale: 0.9, opacity: 0.8 }}
            whileInView={{ scale: 1, opacity: 1 }}
            whileTap={{ scale: 1.1 }}
            transition={{ duration: 1 }}
            onSubmit={handleSubmit}
            className="w-full md:w-1/2 lg:w-1/3 flex flex-col gap-4 border-2 border-black p-5 rounded-lg"
          >
            <input
              type="text"
              value={symptom}
              onChange={(e) => setSymptom(e.target.value)}
              placeholder="Enter a symptom"
              required
              className="p-3 border border-gray-500 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-black"
            />
            <button
              type="submit"
              className="px-4 py-3 text-lg  bg-black border-2 border-black text-white rounded-lg hover:bg-transparent transition-all duration-300 ease-in-out hover:text-black"
            >
              Predict
            </button>
            {load && (
              <div className="flex justify-center mt-4">
                <div className="loader"></div>{" "}
              </div>
            )}

            {!load && recommendation && (
              <div className="flex justify-center items-center mt-4">
                <div className=" text-center text-black p-4 border-2 border-black rounded-lg">
                  <h2 className="text-xl font-semibold mb-2">Predictions:</h2>
                  <p>Disease: {recommendation.disease}</p>
                  <p>Medicine: {recommendation.medicine}</p>
                </div>
              </div>
            )}
          </motion.form>
        </div>
      </section>

      <div id="aboutus" className="p-10 bg-black text-yellow-200">
        <motion.h1
          initial={{ opacity: 0.8, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
          className="text-center text-xl md:text-4xl"
        >
          About us
        </motion.h1>
        <motion.p
          initial={{ opacity: 0.8, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
          className="px-10 py-4"
          style={{
            lineHeight: "1.5rem",
          }}
        >
          This a platform dedicated to enhancing your healthcare experience with
          the power of artificial intelligence. Our mission is to provide
          accessible and personalized medical recommendations, helping users
          find the right medicines for their symptoms without hassle.
        </motion.p>
      </div>

      <div
        id="contactus"
        className="text-center text-lg md:text-3xl py-8 flex flex-col gap-3 flex-wrap bg-yellow-200"
      >
        <motion.h1
          initial={{ opacity: 0.8, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          Contact us
        </motion.h1>
        <motion.p
          initial={{ opacity: 0.8, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          +91-8792692139
        </motion.p>
      </div>
    </>
  );
};

export default MedicineRecommendation;
