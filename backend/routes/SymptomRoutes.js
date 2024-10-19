const express = require("express");
const { getRecommendation } = require("../controllers/symptomController");

const router = express.Router();

router.post("/recommend", getRecommendation);

module.exports = router;
