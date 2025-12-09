# Build stage for frontend
FROM node:18-alpine AS frontend-build
WORKDIR /app/client
COPY client/package*.json ./
RUN npm install
COPY client/ ./
RUN npm run build

# Build stage for backend and final image
FROM node:18-alpine
WORKDIR /app
COPY server/package*.json ./server/
WORKDIR /app/server
RUN npm install --production
COPY server/ ./

# Copy built frontend assets to backend
COPY --from=frontend-build /app/client/dist ../client/dist

EXPOSE 3000
CMD ["node", "index.js"]
