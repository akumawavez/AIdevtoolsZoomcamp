const Client = require("socket.io-client");
const { server, io } = require("../index");

describe("Interview Platform Integration Tests", () => {
    let clientSocket1, clientSocket2;
    let port;

    beforeAll((done) => {
        server.listen(() => {
            port = server.address().port;
            done();
        });
    });

    afterAll((done) => {
        io.close();
        server.close(done);
    });

    beforeEach((done) => {
        clientSocket1 = new Client(`http://localhost:${port}`);
        clientSocket2 = new Client(`http://localhost:${port}`);

        let connectedCount = 0;
        const onConnect = () => {
            connectedCount++;
            if (connectedCount === 2) done();
        };

        clientSocket1.on("connect", onConnect);
        clientSocket2.on("connect", onConnect);
    });

    afterEach(() => {
        clientSocket1.close();
        clientSocket2.close();
    });

    test("should join a room and sync code changes", (done) => {
        const roomId = "test-room-1";
        const newCode = "console.log('Hello World');";

        // Client 2 listens for updates
        clientSocket2.on("code-update", (code) => {
            try {
                expect(code).toBe(newCode);
                done();
            } catch (error) {
                done(error);
            }
        });

        // Client 1 joins room
        clientSocket1.emit("join-room", roomId);

        // Client 2 joins room
        clientSocket2.emit("join-room", roomId);

        // Allow time for joins to propagate
        setTimeout(() => {
            // Client 1 emits code change
            clientSocket1.emit("code-change", { roomId, code: newCode });
        }, 50);
    });

    test("should receive current code when joining an existing session", (done) => {
        const roomId = "test-room-2";
        const initialCode = "const x = 10;";

        // Client 1 joins and sets code
        clientSocket1.emit("join-room", roomId);

        setTimeout(() => {
            clientSocket1.emit("code-change", { roomId, code: initialCode });

            // Client 2 joins later
            setTimeout(() => {
                clientSocket2.on("code-update", (code) => {
                    try {
                        expect(code).toBe(initialCode);
                        done();
                    } catch (e) {
                        done(e);
                    }
                });
                clientSocket2.emit("join-room", roomId);
            }, 50);
        }, 50);
    });
});
