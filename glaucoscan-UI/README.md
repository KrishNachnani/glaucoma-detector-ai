# Glaucoma Detection Project

This is a Next.js application for glaucoma detection using neural networks.

## Environment Variables

This application uses the following environment variables for configuration:

### API Configuration
- `NEXT_PUBLIC_GLAUCOMA_API_URL`: URL for the backend API (default: http://localhost:8900)

### Email Configuration (Contact Form)

The contact form uses [Web3Forms](https://web3forms.com) to send submissions to your email.

Set the following environment variable before running `./run-docker.sh`:

1. Create or open .env.local in glaucoscan-UI/:
```bash
touch .env.local
```

2. Add your Web3Forms Access Key
```bash
NEXT_PUBLIC_WEB3FORMS_ACCESS_KEY=your_web3forms_access_key
```


## Setup

1. Clone the repository
2. Create a `.env` file in the root directory with the required environment variables
3. Install dependencies with `npm install`
4. Run the development server with `npm run dev`

## Docker Deployment

You can deploy the application using Docker:

```bash
# Build and run with default settings
./run-docker.sh
```

## Project Structure

- `/app`: Next.js application routes and pages
- `/components`: React components including UI components and visualizations
- `/public`: Static assets like images