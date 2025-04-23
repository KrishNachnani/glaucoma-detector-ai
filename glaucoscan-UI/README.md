# Glaucoma Detection Project

This is a Next.js application for glaucoma detection using neural networks.

## Environment Variables

This application uses the following environment variables for configuration:

### API Configuration
- `NEXT_PUBLIC_GLAUCOMA_API_URL`: URL for the backend API (default: http://localhost:8900)

### Email Configuration (Contact Form)
- `NEXT_PUBLIC_MAILERSEND_API_KEY`: Your MailerSend API key for sending emails
- `NEXT_PUBLIC_EMAIL_FROM`: Email address used as the sender (default: noreply@yourdomain.com)
- `NEXT_PUBLIC_EMAIL_TO`: Email address where contact form submissions are sent (default: contact@yourdomain.com)

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

# Or with custom environment variables
API_URL=https://your-api-url.com MAILERSEND_API_KEY=your_key EMAIL_FROM=your@email.com EMAIL_TO=recipient@email.com ./run-docker.sh
```

## Project Structure

- `/app`: Next.js application routes and pages
- `/components`: React components including UI components and visualizations
- `/public`: Static assets like images