FROM node:18-alpine

WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Update browserslist database
RUN npx update-browserslist-db@latest

# Copy the rest of the application
COPY . .

# Build the Next.js application
RUN npm run build

# Expose the port the app runs on
EXPOSE 8001

# Command to run the Next.js server on port 8001
CMD ["npm", "run", "start", "--", "-p", "8001"]