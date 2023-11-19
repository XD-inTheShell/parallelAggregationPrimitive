bool loadFromFile(std::string fileName) {
  std::ifstream inFile;
  inFile.open(fileName);
  if (!inFile) {
    return false;
  }

  std::string line;
  while (std::getline(inFile, line)) {
    Particle particle;
    std::stringstream sstream(line);
    std::string str;
    std::getline(sstream, str, ' ');
    particle.mass = (float)atof(str.c_str());
    std::getline(sstream, str, ' ');
    particle.position.x = (float)atof(str.c_str());
    std::getline(sstream, str, ' ');
    particle.position.y = (float)atof(str.c_str());
    std::getline(sstream, str, ' ');
    particle.velocity.x = (float)atof(str.c_str());
    std::getline(sstream, str, '\n');
    particle.velocity.y = (float)atof(str.c_str());
    particle.id = (int)particles.size();
    particles.push_back(particle);
  }
  inFile.close();
  newParticles.resize(particles.size());
  return true;
}
