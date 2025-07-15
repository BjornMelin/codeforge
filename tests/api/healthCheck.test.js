const request = require('supertest');
const app = require('../../app'); // Assuming your main app is exported from app.js

describe('Health Check Endpoint', () => {
    it('should return 200 OK', async () => {
        const res = await request(app).get('/health');
        expect(res.statusCode).toEqual(200);
    });

    it('should return the correct response format', async () => {
        const res = await request(app).get('/health');
        expect(res.body).toEqual({ status: 'UP' });
    });
});