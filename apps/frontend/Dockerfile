#build
FROM node:13-alpine as builder

WORKDIR /app

COPY package.json /app/package.json
COPY package-lock.json /app/package-lock.json

RUN npm install

COPY . /app

RUN npm run build

# serve
FROM nginx:1.16.0-alpine

COPY --from=builder /app/build /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
