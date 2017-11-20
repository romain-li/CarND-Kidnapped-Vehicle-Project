/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <utility>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, const double std[]) {
  // Skip if already initialized
  if (is_initialized)
    return;

  num_particles = 100;

  const double std_x = std[0];
  const double std_y = std[1];
  const double std_theta = std[2];

  default_random_engine generator;
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  for (int i = 0; i < num_particles; ++i) {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(generator);
    particle.y = dist_y(generator);
    particle.theta = dist_theta(generator);
    particle.weight = 1;

    particles.push_back(particle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, const double std_pos[], double velocity, double yaw_rate) {
  const double std_x = std_pos[0];
  const double std_y = std_pos[1];
  const double std_theta = std_pos[2];

  default_random_engine generator;
  normal_distribution<double> dist_x_noise(0, std_x);
  normal_distribution<double> dist_y_noise(0, std_y);
  normal_distribution<double> dist_theta_noise(0, std_theta);

  for (Particle &particle : particles) {
    if (fabs(yaw_rate) >= 0.001) {
      const double theta = particle.theta;
      const double theta_f = theta + yaw_rate * delta_t;
      const double velocity_divided_by_yaw_rate = velocity / yaw_rate;

      particle.x += velocity_divided_by_yaw_rate * (sin(theta_f) - sin(theta)) + dist_x_noise(generator);
      particle.y += velocity_divided_by_yaw_rate * (cos(theta) - cos(theta_f)) + dist_y_noise(generator);
      particle.theta = theta_f + dist_theta_noise(generator);
    } else {
      particle.x += velocity * delta_t + dist_x_noise(generator);
      particle.y += dist_y_noise(generator);
      particle.theta += dist_theta_noise(generator);
    }
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
  /* In this case, there are only 42 landmarks in the map,
   * so we don't need to improve the association by some algorithm.
   * But if we have a huge of landmarks, I will use some KD-Tree and cache to improve the performance:
   * 1. Split the landmarks into many blocks with a suitable width (Greater than double sensor_range, and big enough)
   *      by latitude and longitude.
   * 2. Get the 4 nearby blocks, generate one KD-Tree if not hit the cache.
   * 3. Use KD-Tree to search, the complexity is O(n * log(m))
   */

  for (LandmarkObs &observation: observations) {
    double min_dist = MAXFLOAT;
    for (LandmarkObs landmark: predicted) {
      double tmp_dist = dist(landmark.x, landmark.y, observation.x, observation.y);
      if (tmp_dist < min_dist) {
        min_dist = tmp_dist;
        observation.id = landmark.id;
      }
    }
  }
}

Map::single_landmark_s getLandmark(int id, const Map &map_landmarks) {
  // This is the normal version of landmark finder
  for (Map::single_landmark_s landmark:map_landmarks.landmark_list) {
    if (landmark.id_i == id) {
      return landmark;
    }
  }
  return map_landmarks.landmark_list[0];
  // But in this case, the landmark's id is equals than their offset + 1 like this:
  // return map_landmarks.landmark_list[id - 1];
  // If there are lot of landmarks, we need a map structure or save pointer into observation to improve performance.
}

void ParticleFilter::updateWeights(double sensor_range, const double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  const double sig_x = std_landmark[0];
  const double sig_y = std_landmark[1];
  // Add double std to get more accuracy.
  double dist_threshold = sensor_range + sig_x + sig_y;
  const double gaussian_norm = (1 / (2 * M_PI * sig_x * sig_y));
  const double double_square_sig_x = 2 * sig_x * sig_x;
  const double double_square_sig_y = 2 * sig_y * sig_y;

  weights.clear();

  for (Particle &particle : particles) {
    const double sin_theta = sin(particle.theta);
    const double cos_theta = cos(particle.theta);
    vector<LandmarkObs> observations_transformed;
    vector<LandmarkObs> predicted;

    // Filter map landmarks to observations
    for (Map::single_landmark_s map_landmark : map_landmarks.landmark_list) {
      if (dist(particle.x, particle.y, map_landmark.x_f, map_landmark.y_f) <= dist_threshold) {
        LandmarkObs landmark{};
        landmark.id = map_landmark.id_i;
        landmark.x = map_landmark.x_f;
        landmark.y = map_landmark.y_f;
        predicted.push_back(landmark);
      }
    }

    if (predicted.empty()) {
      // Skip this loop if no landmark in predicted observations.
      return;
    }

    // Transform observations
    for (LandmarkObs observation : observations) {
      LandmarkObs observation_transformed{};
      observation_transformed.x = particle.x + cos_theta * observation.x - sin_theta * observation.y;
      observation_transformed.y = particle.y + sin_theta * observation.x + cos_theta * observation.y;
      observations_transformed.push_back(observation_transformed);
    }

    dataAssociation(predicted, observations_transformed);

    // Calculate the particle weight and set associations.
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;

    particle.weight = 1;
    for (LandmarkObs observation: observations_transformed) {
      Map::single_landmark_s landmark = getLandmark(observation.id, map_landmarks);
      const double weight = gaussian_norm * exp(
          -pow(observation.x - landmark.x_f, 2) / double_square_sig_x -
          pow(observation.y - landmark.y_f, 2) / double_square_sig_y);

      particle.weight *= weight;
      associations.push_back(observation.id);
      sense_x.push_back(static_cast<double>(landmark.x_f));
      sense_y.push_back(static_cast<double>(landmark.y_f));
    }

    SetAssociations(particle, associations, sense_x, sense_y);
    weights.push_back(particle.weight);
  }
}

void ParticleFilter::resample() {
  default_random_engine generator;
  discrete_distribution<unsigned long> distribution(weights.begin(), weights.end());
  vector<Particle> new_particles;
  new_particles.reserve(static_cast<unsigned long>(num_particles));

  for (int i = 0; i < num_particles; ++i) {
    new_particles.push_back(particles[distribution(generator)]);
  }
  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle &particle, std::vector<int> associations, std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = std::move(associations);
  particle.sense_x = std::move(sense_x);
  particle.sense_y = std::move(sense_y);

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
