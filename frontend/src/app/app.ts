import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrls: ['./app.css']
})
export class App {
  userInput = {
    crop: '',
    crop_year: 0,
    season: '',
    state: '',
    area: 0,
    yield: 0,
    method: 'mean'
  };

  apiResult: any = null;

  constructor(private http: HttpClient) {}

  makePrediction() {
    const url = 'https://mlmodel-api-875108721885.asia-south1.run.app/predict';
    
    this.http.post(url, this.userInput).subscribe({
      next: (response) => {
        console.log("Got a response!", response);
        this.apiResult = response; 
      },
      error: (err) => {
        console.error("API Connection Error:", err);
        alert("Could not connect to the ML API server. Check your Cloud URL configuration.");
      }
    });
  }
} 