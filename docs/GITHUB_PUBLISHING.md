# GitHub Publishing Guide

This guide shows how to publish MarineVision Deep Ocean Intelligence to GitHub in a clean, professional way.

## Before you start

Make sure you have:

- a GitHub account
- Git installed locally
- the project folder on your machine

## 1. Initialize Git

Open a terminal in the project root and run:

```bash
git init
git branch -M main
```

## 2. Add the remote repository

Create a new empty GitHub repository named:

```text
MarineVision-Deep-Ocean-Intelligence
```

Then connect it:

```bash
git remote add origin https://github.com/<your-username>/MarineVision-Deep-Ocean-Intelligence.git
```

## 3. Review what will be committed

```bash
git status
git diff --stat
```

## 4. Commit the project

```bash
git add .
git commit -m "Prepare MarineVision for GitHub release"
```

## 5. Push to GitHub

```bash
git push -u origin main
```

If Git asks you to authenticate, sign in with your GitHub credentials or use a personal access token.

## 6. Add the repository description

Use a description like:

> AI-powered underwater object detection and analytics with Streamlit and YOLOv11l.

## 6a. Handle the model checkpoint

Do **not** commit `MarineVision_YOLOv11l_best.pt` to GitHub unless you are using Git LFS.

Recommended approach:

- keep the checkpoint locally in `models/`
- document that users should download it separately
- optionally attach it to a GitHub Release or external storage bucket

## 7. Add a topic list

Suggested GitHub topics:

- streamlit
- computer-vision
- object-detection
- yolo
- underwater-ai
- marine-analytics

## 8. Create a release

After the first push, create a GitHub release such as:

```text
v1.0.0 - Initial public release
```

## 9. Optional GitHub Actions

You can later add CI to:

- run lint checks
- validate imports
- verify the app starts

## Recommended repo checklist

- [x] README with installation and usage
- [x] architecture documentation
- [x] deployment guide
- [x] ignore temporary files
- [x] professional naming
- [ ] GitHub remote configured
- [ ] first commit pushed
