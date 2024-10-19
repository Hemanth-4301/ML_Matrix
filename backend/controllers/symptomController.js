const axios = require("axios");

exports.getRecommendation = async (req, res) => {
  const { symptom } = req.body;

  try {
    const response = await axios.post("http://localhost:5001/predict", {
      symptom,
    });
    res.status(200).json(response.data);
  } catch (error) {
    console.error("Error fetching recommendation:", error.message); // Log the error
    res
      .status(500)
      .json({ message: "Error fetching recommendation", error: error.message });
  }
};
