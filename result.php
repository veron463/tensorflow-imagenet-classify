<?php
require "./vendor/autoload.php";
$url = 'http://192.168.2.166:5000/upload-image';
//$url = 'http://74.208.67.179:5000/upload-image';

// Initialize Guzzle client
$client = new GuzzleHttp\Client();
$filePath = './images/00a1a113-9c5b-4e7c-8409-ef4708c71651.jpg';

// Create a POST request
$response1 = $client->request(
    'POST',
    $url,
    [
        'multipart' => [
            [
                'name'     => 'image',
                'contents' => fopen($filePath, 'r'),
            ],
            [
                'name'     => 'label',
                'contents' => 'British Infantry 1845'
            ]

        ],


    ]
);


echo $response1->getBody();

echo "\r\n";
$response2 = $client->request(
    'POST',
    $url,
    [
        'multipart' => [
            [
                'name'     => 'image',
                'contents' => fopen($filePath, 'r'),
            ],
             [
                'name'     => 'closest_count',
                'contents' => 7,
            ],

        ],


    ]
);

echo "<hr>";
$result = json_decode($response2->getBody());

foreach($result as $NextLabel)
{
        echo "<P>".$NextLabel->label." - ".$NextLabel->score;
}