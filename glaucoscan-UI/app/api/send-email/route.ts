import { NextResponse } from 'next/server';
import { MailerSend, EmailParams, Sender, Recipient } from 'mailersend';

// Environment variables (server-side only)
//const MAILERSEND_API_KEY = process.env.MAILERSEND_API_KEY || '';
const MAILERSEND_API_KEY = 'mlsn.f246cd6923b2043f9ae4730ca7c15f70e1fbd3017d8d8c5fb5f4cacafcb8c870';
const EMAIL_FROM = process.env.EMAIL_FROM || '';
const EMAIL_TO = process.env.EMAIL_TO || '';

export async function POST(request: Request) {
  try {
    // Parse the request body
    const { name, email, message } = await request.json();
    
    // Validate inputs
    if (!name || !email || !message) {
      return NextResponse.json(
        { error: 'Please provide name, email, and message' },
        { status: 400 }
      );
    }
    
    // Validate that API key is available
    if (!MAILERSEND_API_KEY) {
      return NextResponse.json(
        { error: 'Email service is not configured properly' },
        { status: 500 }
      );
    }
    
    // Initialize MailerSend with API key
    const mailerSend = new MailerSend({
      apiKey: MAILERSEND_API_KEY,
    });
    
    // Set up email parameters
    const sentFrom = new Sender(EMAIL_FROM, 'glaucoscan.ai');
    const recipients = [new Recipient(EMAIL_TO, 'Support Team')];
    
    // Create email params
    const emailParams = new EmailParams()
      .setFrom(sentFrom)
      .setTo(recipients)
      .setSubject('New Contact Form Submission')
      .setHtml(`
        <div>
          <h2>New Contact Form Submission</h2>
          <p><strong>Name:</strong> ${name}</p>
          <p><strong>Email:</strong> ${email}</p>
          <p><strong>Message:</strong> ${message}</p>
        </div>
      `)
      .setText(`
        New Contact Form Submission
        
        Name: ${name}
        Email: ${email}
        Message: ${message}
      `);
    
    // Send the email
    const response = await mailerSend.email.send(emailParams);
    
    return NextResponse.json({ success: true, data: response });
  } catch (error) {
    console.error('Email sending error:', error);
    return NextResponse.json(
      { error: 'Failed to send email' },
      { status: 500 }
    );
  }
}