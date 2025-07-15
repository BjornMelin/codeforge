const express = require('express');
const healthCheckRoute = require('./healthCheck');

const router = express.Router();

router.use('/health', healthCheckRoute);

module.exports = router;