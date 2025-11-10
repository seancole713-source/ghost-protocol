#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════╗"
echo "║         Ghost Trading - SSL Setup (Let's Encrypt) ║"
echo "╚════════════════════════════════════════════════════════╝"
echo

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  Please run as root or with sudo"
    exit 1
fi

# Get domain info
read -p "Enter your domain name (e.g., ghost.yourdomain.com): " DOMAIN
read -p "Enter your email for SSL certificate: " EMAIL

echo
echo "🔐 This script will:"
echo "  1. Install Nginx"
echo "  2. Configure reverse proxy"
echo "  3. Install Certbot"
echo "  4. Setup SSL certificate"
echo "  5. Configure auto-renewal"
echo
read -p "Continue? (y/n): " CONTINUE

if [ "$CONTINUE" != "y" ]; then
    echo "Setup cancelled."
    exit 0
fi

# Install Nginx
echo "📦 Installing Nginx..."
apt update
apt install -y nginx

# Stop default site
systemctl stop nginx

# Create Nginx configuration
echo "⚙️  Configuring Nginx..."
cat > /etc/nginx/sites-available/ghost << EOF
server {
    listen 80;
    server_name $DOMAIN;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
EOF

# Enable site
ln -sf /etc/nginx/sites-available/ghost /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test Nginx config
echo "🧪 Testing Nginx configuration..."
nginx -t

# Start Nginx
systemctl start nginx
systemctl enable nginx

# Install Certbot
echo "📦 Installing Certbot..."
apt install -y certbot python3-certbot-nginx

# Obtain SSL certificate
echo "🔒 Obtaining SSL certificate..."
certbot --nginx -d $DOMAIN --non-interactive --agree-tos -m $EMAIL

# Test auto-renewal
echo "🔄 Testing auto-renewal..."
certbot renew --dry-run

echo
echo "╔════════════════════════════════════════════════════════╗"
echo "║              SSL Setup Complete!                   ║"
echo "╚════════════════════════════════════════════════════════╝"
echo
echo "✅ Ghost is now available at: https://$DOMAIN"
echo
echo "📊 Test your setup:"
echo "  curl https://$DOMAIN/health"
echo
echo "🔒 SSL certificate will auto-renew every 90 days"
echo
echo "🔧 Nginx commands:"
echo "  Reload:  systemctl reload nginx"
echo "  Restart: systemctl restart nginx"
echo "  Status:  systemctl status nginx"
echo
